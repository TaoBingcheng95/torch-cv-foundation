import os
import json
import random
import numpy as np
import cv2
import urllib.request as urt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import transforms.voc_transforms as tr
from dataset.utils import encode_segmap, decode_segmap


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 base_dir='pascal',
                 split='train',
                 crop_size = 512
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        """
        super().__init__()

        self.root = base_dir
        self._image_dir = os.path.join(self.root, 'JPEGImages')
        self._ann_dir = os.path.join(self.root, 'SegmentationClass')
        self.split = split
        self.crop_size = crop_size

        self.im_ids = []
        self.images = []
        self.categories = []
        with open(os.path.join(os.path.join(self.root, 'ImageSets', 'Segmentation',  f'{self.split}.txt')), "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                _image = os.path.join(self._image_dir, f"{line}.jpg")
                _cat = os.path.join(self._ann_dir, f"{line}.png")
                if not os.path.isfile(_image):
                    print("Image Not Found: {}".format(_image))
                    continue
                if not os.path.isfile(_cat):
                    print("Category Not Found: {}".format(_cat))
                    continue
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
        assert (len(self.images) == len(self.categories))
        print(f'Number of images in {self.split}: {len(self.images):d}')

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        sample = {'image': _img, 'label': _target}
        # sample = (_img, _target)

        if self.split == "train":
            sample =  self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        img = sample['image']
        label = sample['label']
        return img, label


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


class VocSemanticSegDataSet(Dataset):
    """Build the voc 2007 dataset for segmentation
    """

    def __init__(self, data_file, transformers=None, train_phase=True, voc=False):
        super(VocSemanticSegDataSet, self).__init__()
        if not os.path.isfile(data_file):
            raise TypeError(f"{data_file} must be file type!!!")
        self.data_list = [json.loads(x.strip()) for x in open(data_file).readlines()]
        self.data_indices = [x for x in range(len(self.data_list))]
        self.train_phase = train_phase
        self.voc = voc
        if self.train_phase:
            random.shuffle(self.data_list)

        if transformers is not None:
            self.data_aug = transformers
        else:
            self.data_aug = None

    def _loadImages(self, line):
        img_path = line["image_path"]
        lbl_path = line["label_path"]

        if "http" not in img_path:
            image = cv2.imread(img_path)
            label = cv2.imread(lbl_path)
            # rm 255 border
            if self.voc:
                label = self._rm_border(label)
            else:
                label = self._make_lbl(label)

        # read oss data
        else:
            img_context = urt.urlopen(img_path).read()
            image = cv2.imdecode(np.asarray(bytearray(img_context), dtype='uint8'), cv2.IMREAD_COLOR)

            lbl_context = urt.urlopen(lbl_path).read()
            label = cv2.imdecode(np.asarray(bytearray(lbl_context), dtype='uint8'), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label

    # remove the border 255 from label
    def _rm_border(self, seg):
        pe = np.where(seg == 255)
        seg[pe] = 0
        return seg

        # make value to label

    def _make_lbl(self, seg):
        pe = np.where(seg == 255)
        seg[pe] = 1
        return seg

    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.data_list[index]
                img, lbl = self._loadImages(line)
                if self.data_aug is not None:
                    img, lbl = self.data_aug(img, lbl)
                return img, lbl
            except Exception as e:
                print(f"{self.data_list[index]} have {e} exception!!!")
                index = random.choice(self.data_indices)

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':

    voc_train = VOCSegmentation(base_dir='D:\\myspace\\dataset\\YOLO\\VOC\\VOCdevkit\\VOC2012',
                                split='train')
    # image, label = next(iter(voc_train))
    # segmap = decode_segmap(label, dataset='pascal')
    # fig, axis = plt.subplots(1,2)
    # axis[0].imshow(image.permute(1,2,0))
    # axis[1].imshow(segmap)
    # plt.show()

    dataloader = DataLoader(voc_train, batch_size=4, shuffle=True, num_workers=0)
    x, y = next(iter(dataloader))
    print(x.shape, y.shape)

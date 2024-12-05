import os
import json
import random
import numpy as np
import cv2
import urllib.request as urt
from PIL import Image

import matplotlib.pyplot as plt

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
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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


if __name__ == '__main__':

    voc_train = VOCSegmentation(base_dir='D:\\myspace\\dataset\\VOC\\VOCdevkit\\VOC2012',
                                split='train')

    image, label = next(iter(voc_train))
    print(np.unique(label))
    print(image.shape, label.shape)
    t = image.permute(1,2,0)
    print(t.shape)
    segmap = decode_segmap(label, dataset='pascal')
    fig, axis = plt.subplots(1,2)
    axis[0].imshow(image.permute(1,2,0)) #
    axis[1].imshow(segmap)
    plt.show()

    # dataloader = DataLoader(voc_train, batch_size=4, shuffle=True, num_workers=0)
    # x, y = next(iter(dataloader))
    # print(x.shape, y.shape)

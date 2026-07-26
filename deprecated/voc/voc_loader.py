import os
# import json
# import random
import numpy as np
import cv2
# import urllib.request as urt
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# import transforms.voc_transforms as tr
# from datasets.utils import encode_segmap, decode_segmap


def decode_segmap(label_mask, dataset, plot=False):
    """
    Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = np.zeros_like(label_mask)
    g = np.zeros_like(label_mask)
    b = np.zeros_like(label_mask)
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.size[1], label_mask.size[0], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb




def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])



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

        # if self.split == "train":
        #     sample =  self.transform_tr(sample)
        # elif self.split == 'val':
        #     sample = self.transform_val(sample)
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

    voc_train = VOCSegmentation(base_dir='data/VOCdevkit/VOC2012',
                                split='train')

    image, label = next(iter(voc_train))
    print(image.size, label.size)
    # print(np.unique(label))
    # print(image.shape, label.shape)
    # t = image.permute(1,2,0)
    # print(t.shape)
    segmap = decode_segmap(label, dataset='pascal')
    fig, axis = plt.subplots(1,2)
    axis[0].imshow(image) #
    axis[1].imshow(segmap)
    plt.show()

    # dataloader = DataLoader(voc_train, batch_size=4, shuffle=True, num_workers=0)
    # x, y = next(iter(dataloader))
    # print(x.shape, y.shape)

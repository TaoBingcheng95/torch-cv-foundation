
import os
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2

from torchvision import transforms
from torchvision.datasets import VOCSegmentation

import albumentations as A
from albumentations.pytorch import ToTensorV2

import transforms.voc_transforms as tr
from dataset.utils import encode_segmap, decode_segmap

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

transform_albu = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
    ToTensorV2()
    ],
    p=0.9)

composed_transforms = transforms.Compose([
    tr.RandomHorizontalFlip(),
    tr.RandomScaleCrop(base_size=256, crop_size=256),
    tr.RandomGaussianBlur(),
    # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])


def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data * rgb_std + rgb_mean
    return data.clip(0,1)


def label2image(prelabel):
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w, -1)
    image = np.zeros((h*w,3), dtype = np.uint8)
    for ii in range(21):#共21个类别
        index = np.where(prelabel == ii) # 找到n维数组中特定数值的下标
        image[index,:] = VOC_COLORMAP[ii]  # cmode(ii)
    return image.reshape(h,w,3)


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCSearchDataset(VOCSegmentation):
    def __init__(self, 
                 root="~/data/pascal_voc", 
                 year="2012",
                 image_set="train", 
                 download=False, 
                 transform=None,
                 target_transform=None):
        super().__init__(root=root,
                         year=year,
                         image_set=image_set,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)


    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask


    def __getitem__(self, index):

        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(self.masks[index]))
        mask[mask==255] = 0  # remove the border 255 from label
        # mask = cv2.imread(self.masks[index])
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = self.image2label(mask, VOC_COLORMAP)
        # mask = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


    def label2image(self, prelabel):
        if len(prelabel.sahpe):
            prelabel = prelabel.squeeze()
        h,w = prelabel.shape
        prelabel = prelabel.reshape(h*w, -1)
        image = np.zeros((h*w,3), dtype = np.uint8)
        for ii in range(21): # 共21个类别
            index = np.where(prelabel == ii) # 找到n维数组中特定数值的下标
            image[index,:] = VOC_COLORMAP[ii]
        return image.reshape(h,w,3)


    def image2label(self, image, colormap):
        """
        将颜色转换为类别
        """
        image = np.array(image, dtype = np.uint8)
        cm2lbl = np.zeros(256 * 256 * 256, dtype=np.uint8)
        for label, color in enumerate(colormap):
            # 创建哈希表存储原图颜色序列
            index = color[0] + color[1] * 256 + color[2] * 256 * 256
            cm2lbl[index] = label
        # rgb三通道合并(简单粗暴的三通道相加)
        ix = np.dot(image, [1, 256, 256 * 256])
        # 从哈希表中，将颜色序列转换为对应的标签
        image2 = cm2lbl[ix]
        return image2



if __name__ == '__main__':
    voc2012_dir = Path('D:\myspace\dataset\VOC')
    dataset = PascalVOCSearchDataset(root=voc2012_dir, year='2012', image_set='train', 
                              download=False,
                              # transforms=label_transformer,
                              transform=transform_albu,
                              target_transform=transform_albu,
                              )
    print(len(dataset))
    x, y = next(iter(dataset))
    # x = torch.tensor(np.array(x))
    # y = torch.tensor(np.array(y))
    print(x.shape) # torch.Size([281, 500, 3])
    print(y.shape) # torch.Size([281, 500])
    print(np.unique(y))
    # y_color = decode_segmap(y, 'pascal')

    # 显示图像和分割掩码
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(x.permute(1,2,0)) #.permute(1,2,0)
    ax[0].set_title('Image')
    ax[1].imshow(y)
    ax[1].set_title('Segmentation Mask')
    plt.show()


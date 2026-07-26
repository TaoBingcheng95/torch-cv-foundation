from torchvision import transforms
import torch
import numpy as np
import random
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):
    """随机水平翻转图像和掩码。"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask

class RandomCrop(object):
    """随机裁剪图像和掩码。"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return image, mask

class ToTensor(object):
    """将PIL图像和掩码转换为张量。"""
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

class Normalize(object):
    """归一化图像。"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, self.mean, self.std)
        return image, mask
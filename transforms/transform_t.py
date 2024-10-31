from torchvision import transforms
import torch
import numpy as np
import random

class RandomHorizontalFlip(object):
    """随机水平翻转图像和掩码。"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask

class RandomCrop(object):
    """随机裁剪图像和掩码。"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        return image, mask

class ToTensor(object):
    """将PIL图像和掩码转换为张量。"""
    def __call__(self, image, mask):
        image = transforms.functional.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

class Normalize(object):
    """归一化图像。"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = transforms.functional.normalize(image, self.mean, self.std)
        return image, mask
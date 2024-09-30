import torchvision
import torch.nn as nn

from .submodules import conv2d_bn_relu


class CustomFcn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.custom_module = conv2d_bn_relu(3, 3, 3)
        self.fcn = torchvision.models.segmentation.fcn_resnet50()

    def forward(self, img):
        x = self.custom_module(img)
        y = self.fcn(x)
        return y

class Resnet101Fcn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.custom_module = conv2d_bn_relu(3, 3, 3)
        self.fcn = torchvision.models.segmentation.fcn_resnet101()

    def forward(self, img):
        x = self.custom_module(img)
        y = self.fcn(x)
        return y

class LightFcn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.custom_module = conv2d_bn_relu(3, 3, 3)
        self.fcn = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()

    def forward(self, img):
        x = self.custom_module(img)
        y = self.fcn(x)
        return y

class DeepLabv3Fcn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.custom_module = conv2d_bn_relu(3, 3, 3)
        self.fcn = torchvision.models.segmentation.deeplabv3_resnet50

    def forward(self, img):
        x = self.custom_module(img)
        y = self.fcn(x)
        return y

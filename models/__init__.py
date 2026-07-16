
from .lenet import LeNet5
from .alexnet import AlexNet
from .vgg import VGG16, build_vgg
from .googlenet import build_googlenet
from .resnet import build_resnet

__all__ = ['LeNet5',
           'AlexNet',
           'VGG16',
           'build_vgg',
           'build_googlenet',
           'build_resnet']


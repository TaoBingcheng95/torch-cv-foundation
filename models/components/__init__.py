
from .LeNet import LeNet5, Net
from .AlexNet import AlexNet
from .vit import ViT
from .AttUNet import UNet
from models.components.depredate.resnet import Resnet18
from .BiSeNet import BiSeNetV1
# from .densenet_1 import SimpleDenseNet, SimpleUNet
from .unet import SimpleUNet
from .DenseNet import SimpleDenseNet
# from .SimpleFCN import FCN
from .fcn import FCN
from .DenseNet import DenseNet

__all__ = ['LeNet5',
           'Net',
           'AlexNet',
           'UNet',
           'SimpleDenseNet',
           'SimpleUNet',
           'Resnet18',
           'BiSeNetV1',
           'FCN',
           'DenseNet']


from .LeNet import LeNet5
from .AlexNet import AlexNet
from .vit import ViT
from .AttUNet import UNet
from .BiSeNet import BiSeNetV1
# from .densenet_1 import SimpleDenseNet, SimpleUNet
from .unet import SimpleUNet
from .DenseNet import SimpleDenseNet
# from .SimpleFCN import FCN
from .fcn import FCN
from .DenseNet import DenseNet

__all__ = ['LeNet5',
        #    'Net',
           'AlexNet',
           'UNet',
           'SimpleDenseNet',
           'SimpleUNet',
           'BiSeNetV1',
           'FCN',
           'DenseNet']

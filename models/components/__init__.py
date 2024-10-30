
from .LeNet import LeNet
from .AlexNet import AlexNet
# from .simple_dense_net import SimpleDenseNet
# from .simple_unet import UNet as SimpleUNet
from .vit import ViT
from .attunet import UNet
from .resnet import Resnet18
from .BiSeNet import BiSeNetV1
from .simple_net import ModernDenseNet, SimpleDenseNet, SimpleUNet
from .fcn import FCN

__all__ = ['LeNet',
           'AlexNet',
           'ModernDenseNet',
           'SimpleDenseNet',
           'SimpleUNet',
           'Resnet18',
           'BiSeNetV1',
           'FCN']

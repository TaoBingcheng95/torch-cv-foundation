
# ============ 分类模型 ============
from .lenet import LeNet5
from .alexnet import AlexNet

from .vgg import VGG16, VGG19, build_vgg
from .googlenet import build_googlenet

from .resnet import build_resnet
from .densenet import densenet121, densenet161, densenet169, densenet201

from .mobilenetv2 import mobilenet_v2
from .mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small

from .vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

# ConvNeXt V1 统一从 convnext.py (torchvision 移植版) 导出；
# convnext_official.py 中的同名 V1 工厂仅供模块内教学使用，避免重名冲突
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
from .convnext_official import (
    convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano,
    convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge,
    fcmae_convnextv2_tiny, fcmae_convnextv2_base,
)

# ============ 自监督模型 ============
from .mae import MaskedAutoencoderViT, mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

# ============ 分割模型 ============
from .unet import UNet, UNet_ResNet18, UNet_MobileNetV2
from .fcn import FCN32s, FCN16s, FCN8s, SimpleFCN
from .deeplab3plus import DeepLabV3Plus

__all__ = [
    # LeNet / AlexNet
    'LeNet5',
    'AlexNet',
    # VGG
    'VGG16',
    'VGG19',
    'build_vgg',
    # GoogleNet
    'build_googlenet',
    # ResNet
    'build_resnet',
    # DenseNet
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    # MobileNet
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    # ViT
    'vit_b_16',
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'vit_h_14',
    # ConvNeXt V1
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large',
    # ConvNeXt V2
    'convnextv2_atto',
    'convnextv2_femto',
    'convnextv2_pico',
    'convnextv2_nano',
    'convnextv2_tiny',
    'convnextv2_base',
    'convnextv2_large',
    'convnextv2_huge',
    # FCMAE
    'fcmae_convnextv2_tiny',
    'fcmae_convnextv2_base',
    # MAE
    'MaskedAutoencoderViT',
    'mae_vit_base_patch16',
    'mae_vit_large_patch16',
    'mae_vit_huge_patch14',
    # UNet
    'UNet',
    'UNet_ResNet18',
    'UNet_MobileNetV2',
    # FCN
    'FCN32s',
    'FCN16s',
    'FCN8s',
    'SimpleFCN',
    # DeepLabV3+
    'DeepLabV3Plus',
]

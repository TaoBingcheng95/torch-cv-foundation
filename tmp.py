
import numpy as np
import torch
import torch.nn as nn

from models.deeplab3plus import DeepLabV3Plus
from models.backbone import ResNet18Encoder, VGG16Encoder, MobileNetV2Encoder
# from models.unet import UNet_ResNet18, UNet_MobileNetV2, UNet
from models.convext import convnext_base



if __name__ == '__main__':
    from torchinfo import summary
    
    # 通道数现在由 backbone.out_channels 自动推导，无需手动传入
    # VGG16Encoder()       → low_level=128, high_level=512
    # MobileNetV2Encoder() → low_level=24,  high_level=96
    # ResNet18Encoder()    → low_level=64,  high_level=256
    # model = DeepLabV3Plus(MobileNetV2Encoder())
    model  = convnext_base()

    input_size = (1,3,224,224)
    # dummy_input = torch.randn(input_size)
    # o = model(dummy_input)
    # print(o.shape)

    summary(model, input_size=input_size)

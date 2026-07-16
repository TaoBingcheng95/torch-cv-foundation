
import numpy as np
import torch
import torch.nn as nn

from torchvision.models import VGG, ResNet, DenseNet
from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp



if __name__ == '__main__':
    from torchinfo import summary

    model = smp.Unet(
        encoder_name="resnet34", # 编码器骨干网络
        encoder_weights=None, # 使用ImageNet预训练权重
        in_channels=3, # 输入通道数（RGB为3，灰度图为1）
        classes=2 # 输出类别数
        )
    
    input_size = (1,3,244,244)
    summary(model, input_size=input_size)
            

import os
import torch
import torch.nn.functional as F   # 新增导入
import torch.nn as nn
import torchvision.models as models


class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
        
        # 核心技巧：直接对 nn.Sequential 进行切片！
        self.f0 = mobilenet.features[0:2]   # 1/2, 16ch
        self.f1 = mobilenet.features[2:4]   # 1/4, 24ch
        self.f2 = mobilenet.features[4:7]   # 1/8, 32ch
        self.f3 = mobilenet.features[7:14]  # 1/16, 96ch
        self.f4 = mobilenet.features[14:18] # 1/32, 320ch

    def forward(self, x):
        f0 = self.f0(x)
        f1 = self.f1(f0)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = self.f4(f3)
        return [f0, f1, f2, f3, f4]

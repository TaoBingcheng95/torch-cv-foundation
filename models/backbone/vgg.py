"""
VGG Backbone — 特征提取组件
=============================

将 VGG 的 5 个卷积阶段拆解为独立的 block1 ~ block5，
每个 block 可单独访问，便于：
  1. 观察每一层的特征图变化（教学演示）
  2. 作为分割/检测等任务的 Encoder

forward() 返回一个字典，包含每个 block 的输出特征图，
下游任务可按需取用不同层级的特征。

输入尺寸: 3×224×224
各 block 输出尺寸（以 224×224 输入为例）：
    block1: 64 通道,  112×112  (2×3×3 卷积 + 池化)
    block2: 128 通道, 56×56    (2×3×3 卷积 + 池化)
    block3: 256 通道, 28×28    (3×3×3 卷积 + 池化, VGG19 为 4×3×3)
    block4: 512 通道, 14×14    (3×3×3 卷积 + 池化, VGG19 为 4×3×3)
    block5: 512 通道, 7×7      (3×3×3 卷积 + 池化, VGG19 为 4×3×3)
"""

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, VGG19_Weights



class VGG16Encoder(nn.Module):
    """
    VGG16 特征提取器 (Backbone)。

    将 VGG16 的 5 个卷积阶段拆分为 block1 ~ block5，
    forward() 返回字典 {'block1': feat, ..., 'block5': feat}，
    方便下游任务按需取用不同层级的特征。
    """
    out_channels = [64, 128, 256, 512, 512]
    def __init__(self, weights=False):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/2

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/4

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/8

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/16

            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/32
        )
        if weights:
            self.load_state_dict(VGG16_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True), 
                                strict=False)

        self.split_features()


    def split_features(self):
        self.block1 = nn.Sequential(*list(self.features.children())[:5])
        self.block2 = nn.Sequential(*list(self.features.children())[5:10])
        self.block3 = nn.Sequential(*list(self.features.children())[10:17])
        self.block4 = nn.Sequential(*list(self.features.children())[17:24])
        self.block5 = nn.Sequential(*list(self.features.children())[24:])


    def forward(self, x):
        # 使用字典存储每一层的输出，Key 为 block 名称，Value 为特征图
        features = {}        
        x = self.block1(x)
        features['block1'] = x  # Shape: [B, 64, H/2, W/2]
        x = self.block2(x)
        features['block2'] = x  # Shape: [B, 128, H/4, W/4]
        x = self.block3(x)
        features['block3'] = x  # Shape: [B, 256, H/8, W/8]
        x = self.block4(x)
        features['block4'] = x  # Shape: [B, 512, H/16, W/16]
        x = self.block5(x)
        features['block5'] = x  # Shape: [B, 512, H/32, W/32]
        return features



class VGG19Encoder(nn.Module):
    """VGG19 特征提取器 (Backbone)。

    与 VGG16 的区别：Block 3/4/5 各多一层 3×3 卷积（4 层卷积 + 1 层池化）。
    forward() 返回字典 {'block1': feat, ..., 'block5': feat}。
    """
    out_channels = [64, 128, 256, 512, 512]
    def __init__(self, weights=False):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1: 2 conv + pool = 5 layers
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/2

            # Conv Block 2: 2 conv + pool = 5 layers
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4

            # Conv Block 3: 4 conv + pool = 9 layers (VGG16 为 3 conv)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/8

            # Conv Block 4: 4 conv + pool = 9 layers (VGG16 为 3 conv)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/16

            # Conv Block 5: 4 conv + pool = 9 layers (VGG16 为 3 conv)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/32
        )
        if weights:
            self.load_state_dict(VGG19_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True),
                                strict=False)

        self.split_features()

    def split_features(self):
        # VGG19: Block1=5, Block2=5, Block3=9, Block4=9, Block5=9
        self.block1 = nn.Sequential(*list(self.features.children())[:5])
        self.block2 = nn.Sequential(*list(self.features.children())[5:10])
        self.block3 = nn.Sequential(*list(self.features.children())[10:19])
        self.block4 = nn.Sequential(*list(self.features.children())[19:28])
        self.block5 = nn.Sequential(*list(self.features.children())[28:])

    def forward(self, x):
        features = {}
        x = self.block1(x)
        features['block1'] = x  # Shape: [B, 64, H/2, W/2]
        x = self.block2(x)
        features['block2'] = x  # Shape: [B, 128, H/4, W/4]
        x = self.block3(x)
        features['block3'] = x  # Shape: [B, 256, H/8, W/8]
        x = self.block4(x)
        features['block4'] = x  # Shape: [B, 512, H/16, W/16]
        x = self.block5(x)
        features['block5'] = x  # Shape: [B, 512, H/32, W/32]
        return features



if __name__ == "__main__":
    model = VGG16Encoder()
    dummy_input = torch.randn(1, 3, 224, 224)
    out_features = model(dummy_input)
    print(type(out_features))
    
    for name, feat in out_features.items():
        print(f"{name} shape: {feat.shape}")

"""
AlexNet Backbone — 特征提取组件
================================

将 AlexNet 的 5 个卷积层拆解为独立的 block1 ~ block5，
每个 block 可单独访问，便于：
  1. 观察每一层的特征图变化（教学演示）
  2. 作为分割/检测等任务的 Encoder（如 DeepLabV3+）

forward() 返回一个字典，包含每个 block 的输出特征图，
下游任务可按需取用不同层级的特征。

输入尺寸: 3×224×224
各 block 输出尺寸（以 224×224 输入为例）：
    block1: 64 通道,  27×27   (11×11 卷积 stride=4 + 池化)
    block2: 192 通道, 13×13   (5×5 卷积 + 池化)
    block3: 384 通道, 13×13   (3×3 卷积，无池化)
    block4: 256 通道, 13×13   (3×3 卷积，无池化)
    block5: 256 通道, 6×6     (3×3 卷积 + 池化)
"""

import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights



class AlexEncoder(nn.Module):
    """
    AlexNet 特征提取器 (Backbone)。

    将 AlexNet 的 5 个卷积阶段拆分为 block1 ~ block5，
    forward() 返回字典 {'block1': feat, ..., 'block5': feat}，
    方便下游任务（如分割、检测）按需取用不同层级的特征。
    """
    out_channels = [64, 192, 384, 256, 256]
    def __init__(self, weights=False) -> None:
        super().__init__()

        # ============================================================
        # 完整的特征提取层（用于 split_features 拆分）
        # ============================================================
        self.features = nn.Sequential(
            # Block 1: C1 + Pool
            #   Conv: 3→64, 11×11, stride=4, padding=2
            #         224×224 → 55×55  ((224+4-11)/4+1 = 55)
            #   Pool: 3×3, stride=2 → 27×27
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   # 0
            nn.ReLU(inplace=True),                                    # 1
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 2

            # Block 2: C2 + Pool
            #   Conv: 64→192, 5×5, stride=1, padding=2
            #         27×27 → 27×27  ((27+4-5)/1+1 = 27)
            #   Pool: 3×3, stride=2 → 13×13
            nn.Conv2d(64, 192, kernel_size=5, padding=2),             # 3
            nn.ReLU(inplace=True),                                    # 4
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 5

            # Block 3: C3（无池化）
            #   Conv: 192→384, 3×3, padding=1 → 13×13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),            # 6
            nn.ReLU(inplace=True),                                    # 7

            # Block 4: C4（无池化）
            #   Conv: 384→256, 3×3, padding=1 → 13×13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),            # 8
            nn.ReLU(inplace=True),                                    # 9

            # Block 5: C5 + Pool
            #   Conv: 256→256, 3×3, padding=1 → 13×13
            #   Pool: 3×3, stride=2 → 6×6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),            # 10
            nn.ReLU(inplace=True),                                    # 11
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 12
        )
        if weights:
            self.load_state_dict(AlexNet_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True), 
                                strict=False)

        # ============================================================
        # 将 features 拆分为 5 个独立 block
        # 拆分后 block1~block5 与 features 共享同一组参数（引用同一对象），
        # forward 中通过 block1~block5 逐层执行，以便收集每层的输出特征。
        # ============================================================
        self.block1: nn.Sequential = None  # C1 + Pool  (index 0-2)
        self.block2: nn.Sequential = None  # C2 + Pool  (index 3-5)
        self.block3: nn.Sequential = None  # C3         (index 6-7)
        self.block4: nn.Sequential = None  # C4         (index 8-9)
        self.block5: nn.Sequential = None  # C5 + Pool  (index 10-12)
        self.split_features()

    def split_features(self):
        """将 self.features 按卷积块拆分为 block1 ~ block5"""
        children = list(self.features.children())
        self.block1 = nn.Sequential(*children[0:3])   # Conv + ReLU + Pool
        self.block2 = nn.Sequential(*children[3:6])   # Conv + ReLU + Pool
        self.block3 = nn.Sequential(*children[6:8])   # Conv + ReLU
        self.block4 = nn.Sequential(*children[8:10])  # Conv + ReLU
        self.block5 = nn.Sequential(*children[10:13]) # Conv + ReLU + Pool

    def forward(self, x: torch.Tensor) -> dict:
        """
        逐 block 提取特征，返回字典。

        维度变化（输入 3×224×224）：
            block1: (B, 64, 27, 27)   — 大卷积核快速降采样
            block2: (B, 192, 13, 13)  — 中等卷积核 + 池化
            block3: (B, 384, 13, 13)  — 小卷积核，保持分辨率
            block4: (B, 256, 13, 13)  — 小卷积核，保持分辨率
            block5: (B, 256, 6, 6)    — 小卷积核 + 池化
        """
        features = {}
        x = self.block1(x)
        features['block1'] = x   # (B, 64, 27, 27)
        x = self.block2(x)
        features['block2'] = x   # (B, 192, 13, 13)
        x = self.block3(x)
        features['block3'] = x   # (B, 384, 13, 13)
        x = self.block4(x)
        features['block4'] = x   # (B, 256, 13, 13)
        x = self.block5(x)
        features['block5'] = x   # (B, 256, 6, 6)
        return features



if __name__ == "__main__":
    model = AlexEncoder(weights=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    out_features = model(dummy_input)

    print(f"返回类型: {type(out_features)}\n")
    for name, feat in out_features.items():
        print(f"{name} shape: {feat.shape}")

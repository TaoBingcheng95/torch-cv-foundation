import os
import torch
import torch.nn.functional as F   # 新增导入
import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights



class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 多尺度特征提取器。

    torchvision mobilenet_v2 features 共 18 层 (index 0~17):
        [0]   Stem Conv3x3, stride=2  → 1/2,  32ch
        [1]   IR(32→16),  stride=1    → 1/2,  16ch
        [2]   IR(16→24),  stride=2    → 1/4,  24ch   (C1)
        [3]   IR(24→24),  stride=1    → 1/4,  24ch
        [4]   IR(24→32),  stride=2    → 1/8,  32ch   (C2)
        [5]   IR(32→32),  stride=1    → 1/8,  32ch
        [6]   IR(32→32),  stride=1    → 1/8,  32ch
        [7]   IR(32→64),  stride=2    → 1/16, 64ch   (C3)
        [8]   IR(64→64),  stride=1    → 1/16, 64ch
        [9]   IR(64→64),  stride=1    → 1/16, 64ch
        [10]  IR(64→64),  stride=1    → 1/16, 64ch
        [11]  IR(64→96),  stride=1    → 1/16, 96ch
        [12]  IR(96→96),  stride=1    → 1/16, 96ch
        [13]  IR(96→96),  stride=1    → 1/16, 96ch
        [14]  IR(96→160), stride=2    → 1/32, 160ch  (C4)
        [15]  IR(160→160),stride=1    → 1/32, 160ch
        [16]  IR(160→160),stride=1    → 1/32, 160ch
        [17]  IR(160→320),stride=1    → 1/32, 320ch

    输出 5 个尺度的特征图:
        block1: 1/2,  16ch
        block2: 1/4,  24ch
        block3: 1/8,  32ch
        block4: 1/16, 96ch
        block5: 1/32, 320ch
    """
    out_channels = [16, 24, 32, 96, 320]

    def __init__(self, weights=None):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if weights else None
        mobilenet = mobilenet_v2(weights=weights)

        # 核心技巧：直接对 nn.Sequential 进行切片！
        self.f0 = mobilenet.features[0:2]   # 1/2,  16ch (Stem + IR→16)
        self.f1 = mobilenet.features[2:4]   # 1/4,  24ch
        self.f2 = mobilenet.features[4:7]   # 1/8,  32ch
        self.f3 = mobilenet.features[7:14]  # 1/16, 96ch
        self.f4 = mobilenet.features[14:18] # 1/32, 320ch

    def forward(self, x):
        features = {}
        f0 = self.f0(x)
        features['block1'] = f0
        f1 = self.f1(f0)
        features['block2'] = f1
        f2 = self.f2(f1)
        features['block3'] = f2
        f3 = self.f3(f2)
        features['block4'] = f3
        f4 = self.f4(f3)
        features['block5'] = f4
        return features


# ---------------------------------------------------------------------------
# MobileNetV3 Backbone
# ---------------------------------------------------------------------------
# MobileNetV3 相比 V2 的主要升级：
#   1. 引入 SE 注意力模块 (Squeeze-and-Excitation)
#   2. 激活函数由 ReLU6 替换为 h-swish
#   3. 网络结构由 NAS 搜索得到，分为 Large / Small 两个版本
#
# torchvision 的 mobilenet_v3 的 features 结构：
#   [0]  Stem (Conv3x3, stride=2)
#   [1..N-1]  InvertedResidual blocks
#   [N-1]  Final 1x1 Conv (project to high-dim)
# ---------------------------------------------------------------------------


class MobileNetV3LargeEncoder(nn.Module):
    """
    MobileNetV3-Large 多尺度特征提取器。

    torchvision mobilenet_v3_large features 共 18 层 (index 0~17):
        [0]   Stem, stride=2          → 1/2,   16ch
        [1]   IR(16→16),  stride=1    → 1/2
        [2]   IR(16→24),  stride=2    → 1/4    (C1)
        [3]   IR(24→24),  stride=1    → 1/4
        [4]   IR(24→40),  stride=2    → 1/8    (C2)
        [5]   IR(40→40),  stride=1    → 1/8
        [6]   IR(40→40),  stride=1    → 1/8
        [7]   IR(40→80),  stride=2    → 1/16   (C3)
        [8]   IR(80→80),  stride=1    → 1/16
        [9]   IR(80→80),  stride=1    → 1/16
        [10]  IR(80→80),  stride=1    → 1/16
        [11]  IR(80→112), stride=1    → 1/16
        [12]  IR(112→112),stride=1    → 1/16
        [13]  IR(112→160),stride=2    → 1/32   (C4)
        [14]  IR(160→160),stride=1    → 1/32
        [15]  IR(160→160),stride=1    → 1/32
        [16]  Conv1x1 (160→960)       → 1/32   (project)

    输出 5 个尺度的特征图:
        block1: 1/2,   16ch
        block2: 1/4,   24ch
        block3: 1/8,   40ch
        block4: 1/16, 112ch
        block5: 1/32, 960ch
    """
    out_channels = [16, 24, 40, 112, 960]

    def __init__(self, weights=None):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if weights else None
        mobilenet = mobilenet_v3_large(weights=weights)

        self.f0 = mobilenet.features[0:2]    # 1/2,   16ch
        self.f1 = mobilenet.features[2:4]    # 1/4,   24ch
        self.f2 = mobilenet.features[4:7]    # 1/8,   40ch
        self.f3 = mobilenet.features[7:13]   # 1/16, 112ch
        self.f4 = mobilenet.features[13:17]  # 1/32, 960ch (含最终 1x1 conv)

    def forward(self, x):
        features = {}
        f0 = self.f0(x)
        features['block1'] = f0
        f1 = self.f1(f0)
        features['block2'] = f1
        f2 = self.f2(f1)
        features['block3'] = f2
        f3 = self.f3(f2)
        features['block4'] = f3
        f4 = self.f4(f3)
        features['block5'] = f4
        return features


class MobileNetV3SmallEncoder(nn.Module):
    """
    MobileNetV3-Small 多尺度特征提取器。

    torchvision mobilenet_v3_small features 共 13 层 (index 0~12):
        [0]   Stem, stride=2          → 1/2,   16ch
        [1]   IR(16→16),  stride=2    → 1/4    (C1, SE)
        [2]   IR(16→24),  stride=2    → 1/8    (C2)
        [3]   IR(24→24),  stride=1    → 1/8
        [4]   IR(24→40),  stride=2    → 1/16   (C3, SE)
        [5]   IR(40→40),  stride=1    → 1/16   (SE)
        [6]   IR(40→40),  stride=1    → 1/16   (SE)
        [7]   IR(40→48),  stride=1    → 1/16   (SE)
        [8]   IR(48→48),  stride=1    → 1/16   (SE)
        [9]   IR(48→96),  stride=2    → 1/32   (C4, SE)
        [10]  IR(96→96),  stride=1    → 1/32   (SE)
        [11]  IR(96→96),  stride=1    → 1/32   (SE)
        [12]  Conv1x1 (96→576)        → 1/32   (project)

    输出 5 个尺度的特征图:
        block1: 1/2,   16ch
        block2: 1/4,   16ch
        block3: 1/8,   24ch
        block4: 1/16,  48ch
        block5: 1/32, 576ch
    """
    out_channels = [16, 16, 24, 48, 576]

    def __init__(self, weights=None):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if weights else None
        mobilenet = mobilenet_v3_small(weights=weights)

        self.f0 = mobilenet.features[0:2]    # 1/2,   16ch
        self.f1 = mobilenet.features[2:3]    # 1/4,   16ch
        self.f2 = mobilenet.features[3:5]    # 1/8,   24ch
        self.f3 = mobilenet.features[5:9]    # 1/16,  48ch
        self.f4 = mobilenet.features[9:13]   # 1/32, 576ch (含最终 1x1 conv)

    def forward(self, x):
        features = {}
        f0 = self.f0(x)
        features['block1'] = f0
        f1 = self.f1(f0)
        features['block2'] = f1
        f2 = self.f2(f1)
        features['block3'] = f2
        f3 = self.f3(f2)
        features['block4'] = f3
        f4 = self.f4(f3)
        return features

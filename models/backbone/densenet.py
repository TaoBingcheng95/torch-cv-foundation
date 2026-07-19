
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import densenet121, densenet169, DenseNet121_Weights, DenseNet169_Weights



class Densenet121Encoder(nn.Module):
    """
    DenseNet121 多尺度特征提取器 (Backbone)。

    torchvision densenet121 的 features 共 12 层 (index 0~11):
        [0]   conv0: Conv2d(3→64, 7×7, stride=2)   → 1/2,  64ch
        [1]   norm0: BatchNorm2d(64)
        [2]   relu0: ReLU
        [3]   pool0: MaxPool2d(3, stride=2)         → 1/4
        [4]   denseblock1: 6 layers, 64→128ch
        [5]   transition1: BN+ReLU+Conv1x1(128→128)+AvgPool  → 1/8,  128ch
        [6]   denseblock2: 12 layers, 128→256ch
        [7]   transition2: BN+ReLU+Conv1x1(256→256)+AvgPool  → 1/16, 256ch
        [8]   denseblock3: 24 layers, 256→512ch
        [9]   transition3: BN+ReLU+Conv1x1(512→512)+AvgPool  → 1/32, 512ch
        [10]  denseblock4: 16 layers, 512→1024ch
        [11]  norm5: BatchNorm2d(1024)

    按 transition 下采样边界切分为 5 级，输出字典:
        block1: 1/2,   64ch   (Stem: conv+bn+relu)
        block2: 1/4,  128ch   (pool + denseblock1 + transition1)
        block3: 1/8,  256ch   (denseblock2 + transition2)
        block4: 1/16, 512ch   (denseblock3 + transition3)
        block5: 1/32, 1024ch  (denseblock4 + final bn + relu)
    """
    out_channels = [64, 128, 256, 512, 1024]

    def __init__(self, weights=None, progress=True, **kwargs):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if weights else None
        backbone = densenet121(weights=weights, progress=progress, **kwargs)

        # 按 transition 下采样边界切分 features
        self.layer0 = backbone.features[0:3]    # 1/2,   64ch  (conv0+norm0+relu0)
        self.layer1 = backbone.features[3:6]    # 1/4,  128ch  (pool0+denseblock1+transition1)
        self.layer2 = backbone.features[6:8]    # 1/8,  256ch  (denseblock2+transition2)
        self.layer3 = backbone.features[8:10]   # 1/16, 512ch  (denseblock3+transition3)
        self.layer4 = backbone.features[10:12]  # 1/32, 1024ch (denseblock4+norm5)

    def forward(self, x):
        features = {}
        x = self.layer0(x)
        features['block1'] = x          # [B, 64, H/2, W/2]
        x = self.layer1(x)
        features['block2'] = x          # [B, 128, H/4, W/4]
        x = self.layer2(x)
        features['block3'] = x          # [B, 256, H/8, W/8]
        x = self.layer3(x)
        features['block4'] = x          # [B, 512, H/16, W/16]
        x = self.layer4(x)
        features['block5'] = F.relu(x, inplace=True)  # [B, 1024, H/32, W/32]
        return features


class Densenet169Encoder(nn.Module):
    """
    DenseNet169 多尺度特征提取器 (Backbone)。

    torchvision densenet169 的 features 共 12 层 (index 0~11)，结构与 121 相同，
    区别在于各 denseblock 的层数更多，最终通道数为 1664。

    按 transition 下采样边界切分为 5 级，输出字典:
        block1: 1/2,   64ch   (Stem: conv+bn+relu)
        block2: 1/4,  128ch   (pool + denseblock1 + transition1)
        block3: 1/8,  256ch   (denseblock2 + transition2)
        block4: 1/16, 512ch   (denseblock3 + transition3)
        block5: 1/32, 1664ch  (denseblock4 + final bn + relu)
    """
    out_channels = [64, 128, 256, 512, 1664]

    def __init__(self, weights=None, progress=True, **kwargs):
        super().__init__()
        weights = DenseNet169_Weights.DEFAULT if weights else None
        backbone = densenet169(weights=weights, progress=progress, **kwargs)

        # 切分方式与 DenseNet121 完全一致
        self.layer0 = backbone.features[0:3]    # 1/2,   64ch
        self.layer1 = backbone.features[3:6]    # 1/4,  128ch
        self.layer2 = backbone.features[6:8]    # 1/8,  256ch
        self.layer3 = backbone.features[8:10]   # 1/16, 512ch
        self.layer4 = backbone.features[10:12]  # 1/32, 1664ch

    def forward(self, x):
        features = {}
        x = self.layer0(x)
        features['block1'] = x          # [B, 64, H/2, W/2]
        x = self.layer1(x)
        features['block2'] = x          # [B, 128, H/4, W/4]
        x = self.layer2(x)
        features['block3'] = x          # [B, 256, H/8, W/8]
        x = self.layer3(x)
        features['block4'] = x          # [B, 512, H/16, W/16]
        x = self.layer4(x)
        features['block5'] = F.relu(x, inplace=True)  # [B, 1664, H/32, W/32]
        return features



if __name__ == "__main__":
    from torchinfo import summary
    backbone = densenet121(weights=None, progress=False)
    summary(backbone, input_size=(1, 3, 224, 224))

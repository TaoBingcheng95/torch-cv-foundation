
import torch
import torch.nn as nn
import torchvision.models as models



class ResNet18Encoder(nn.Module):
    """
    ResNet18 特征提取器 (Backbone)，用于 U-Net 等需要多尺度特征的架构。

    基于 torchvision 的 ResNet18，拆解为 layer0~layer4 五个阶段，
    forward() 返回字典 {'block1': feat, ..., 'block5': feat}，方便 Decoder 按 key 取用。

    各阶段输出（以 3×224×224 输入为例）：
        block1: 64 通道,  112×112  (conv 7×7 stride=2)
        block2: 64 通道,  56×56    (maxpool + BasicBlock×2)
        block3: 128 通道, 28×28    (BasicBlock×2, stride=2)
        block4: 256 通道, 14×14    (BasicBlock×2, stride=2)
        block5: 512 通道, 7×7      (BasicBlock×2, stride=2)
    """
    out_channels = [64, 64, 128, 256, 512]
    def __init__(self, weights=None):
        super().__init__()
        # 加载官方预训练模型
        weights = models.ResNet18_Weights.DEFAULT if weights else None
        resnet = models.resnet18(weights=weights)

        # 拆解 ResNet 为 5 个阶段
        # layer0: Stem 的卷积+BN+ReLU（不含 maxpool），输出 1/2 尺寸
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # layer1: maxpool + 第一组残差块，输出 1/4 尺寸
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # layer2~4: 直接引用原模型的残差层
        self.layer2 = resnet.layer2  # 输出通道: 128, 尺寸 1/8
        self.layer3 = resnet.layer3  # 输出通道: 256, 尺寸 1/16
        self.layer4 = resnet.layer4  # 输出通道: 512, 尺寸 1/32

    def forward(self, x):
        # 使用字典存储每一层的输出，Key 为 layer 名称，Value 为特征图
        features = {}
        x = self.layer0(x)
        features['block1'] = x  # Shape: [B, 64, H/2, W/2]
        x = self.layer1(x)
        features['block2'] = x  # Shape: [B, 64, H/4, W/4]
        x = self.layer2(x)
        features['block3'] = x  # Shape: [B, 128, H/8, W/8]
        x = self.layer3(x)
        features['block4'] = x  # Shape: [B, 256, H/16, W/16]
        x = self.layer4(x)
        features['block5'] = x  # Shape: [B, 512, H/32, W/32]
        return features



class ResNet50Encoder(nn.Module):
    """
    ResNet50 特征提取器 (Backbone)，用于 U-Net 等需要多尺度特征的架构。

    基于 torchvision 的 ResNet50，拆解为 layer0~layer4 五个阶段。
    注意：ResNet50 使用 Bottleneck，输出通道 = planes × 4。
    forward() 返回字典 {'block1': feat, ..., 'block5': feat}，方便 Decoder 按 key 取用。

    各阶段输出（以 3×224×224 输入为例）：
        block1: 64 通道,   112×112  (conv 7×7 stride=2)
        block2: 256 通道,  56×56    (maxpool + Bottleneck×3)
        block3: 512 通道,  28×28    (Bottleneck×4, stride=2)
        block4: 1024 通道, 14×14    (Bottleneck×6, stride=2)
        block5: 2048 通道, 7×7      (Bottleneck×3, stride=2)
    """
    out_channels = [64, 256, 512, 1024, 2048]
    def __init__(self, weights=None):
        super().__init__()
        # 加载官方预训练模型
        weights = models.ResNet50_Weights.DEFAULT if weights else None
        resnet = models.resnet50(weights=weights)

        # 拆解 ResNet 为 5 个阶段
        # layer0: Stem 的卷积+BN+ReLU（不含 maxpool），输出 1/2 尺寸
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # layer1: maxpool + 第一组残差块，输出 1/4 尺寸
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # layer2~4: 直接引用原模型的残差层
        self.layer2 = resnet.layer2  # 输出通道: 512, 尺寸 1/8
        self.layer3 = resnet.layer3  # 输出通道: 1024, 尺寸 1/16
        self.layer4 = resnet.layer4  # 输出通道: 2048, 尺寸 1/32

    def forward(self, x):
        # 使用字典存储每一层的输出，Key 为 layer 名称，Value 为特征图
        features = {}
        x = self.layer0(x)
        features['block1'] = x  # Shape: [B, 64, H/2, W/2]
        x = self.layer1(x)
        features['block2'] = x  # Shape: [B, 256, H/4, W/4]
        x = self.layer2(x)
        features['block3'] = x  # Shape: [B, 512, H/8, W/8]
        x = self.layer3(x)
        features['block4'] = x  # Shape: [B, 1024, H/16, W/16]
        x = self.layer4(x)
        features['block5'] = x  # Shape: [B, 2048, H/32, W/32]
        return features

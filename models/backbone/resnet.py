
import os
import torch
import torch.nn.functional as F   # 新增导入
import torch.nn as nn
import torchvision.models as models



class ResNet18Encoder(nn.Module):
    """
    将 torchvision 的 ResNet18 改造为 U-Net 的 Encoder (Backbone)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载官方预训练模型 (如果 pretrained=True)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # 拆解并重组需要的部分 (截断网络)
        # Initial 阶段：包含第一次下采样 (stride=2) 和 MaxPool (stride=2)，总共缩小 4 倍
        # self.initial = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool
        # )

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # stride=2, 尺寸 1/2, 64ch
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)         # stride=2, 尺寸 1/4,
        # Layer 2 ~ 4：直接引用原模型的层
        # self.layer1 = resnet.layer1  # 输出通道: 64
        self.layer2 = resnet.layer2  # 输出通道: 128
        self.layer3 = resnet.layer3  # 输出通道: 256
        self.layer4 = resnet.layer4  # 输出通道: 512

    def forward(self, x):
        # 逐层提取特征，并保存下来用于后续的 Skip Connection
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # 返回一个列表，方便 U-Net 的 Decoder 按索引取用
        return [x0, x1, x2, x3, x4]



class ResNet50Encoder(nn.Module):
    """
    将 torchvision 的 ResNet18 改造为 U-Net 的 Encoder (Backbone)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载官方预训练模型 (如果 pretrained=True)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # 拆解并重组需要的部分 (截断网络)
        # Initial 阶段：包含第一次下采样 (stride=2) 和 MaxPool (stride=2)，总共缩小 4 倍
        # self.initial = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool
        # )

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # stride=2, 尺寸 1/2, 64ch
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)         # stride=2, 尺寸 1/4,
        # Layer 2 ~ 4：直接引用原模型的层
        # self.layer1 = resnet.layer1  # 输出通道: 64
        self.layer2 = resnet.layer2  # 输出通道: 128
        self.layer3 = resnet.layer3  # 输出通道: 256
        self.layer4 = resnet.layer4  # 输出通道: 512

    def forward(self, x):
        # 逐层提取特征，并保存下来用于后续的 Skip Connection
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # 返回一个列表，方便 U-Net 的 Decoder 按索引取用
        return [x0, x1, x2, x3, x4]


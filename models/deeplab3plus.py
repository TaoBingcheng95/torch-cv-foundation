import torch
from torch import nn
import torch.nn.functional as F

from .neck import ASPP



class Decoder(nn.Module):
    """
    DeepLabV3+ Decoder 模块

    将 ASPP 输出与低层特征融合：
      1. 低层特征 1x1 卷积降维至 48 通道
      2. ASPP 输出上采样至低层特征分辨率
      3. 拼接后两轮 3x3 卷积细化
      4. 1x1 卷积输出类别预测图
    """
    def __init__(self, num_classes, low_level_channels=64, aspp_out_channels=256):
        super().__init__()
        # 低层特征降维至 48 通道
        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        # 特征融合与细化 (aspp_out_channels + 48 通道)
        self.conv1 = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 分类预测头 (Head)
        self.conv2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, encoder_output, low_level_features):
        size = low_level_features.shape[2:]

        # 将 ASPP 输出上采样到与低层特征相同的分辨率
        encoder_output = F.interpolate(encoder_output,
                                       size=size,
                                       mode='bilinear', align_corners=True)

        # 处理低层特征 1x1 降维
        low_level_features = self.low_level_project(low_level_features)

        # 拼接并细化
        x = self.conv1(torch.cat([encoder_output, low_level_features], dim=1))

        # 输出类别预测图
        x = self.conv2(x)
        return x



class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 语义分割模型

    结构: Backbone → ASPP (Neck) → Decoder → 上采样

    参数:
        backbone: 特征提取器，forward 返回字典，需具备 out_channels 类属性，
                  且字典需包含 'block2' (低层, 1/4) 和 'block4' (高层, 1/16) 两个 key
        num_classes: 分割类别数
        aspp_out_channels: ASPP 各分支输出通道数 (默认 256，主流实现标准值)
    """
    def __init__(self, backbone,
                 num_classes=21,
                 aspp_out_channels=256):
        super().__init__()

        self.backbone = backbone

        # 从 backbone 自动读取各 block 输出通道数
        out_channels = backbone.out_channels
        low_level_channels = out_channels[1]   # block2: 1/4 分辨率
        high_level_channels = out_channels[3]  # block4: 1/16 分辨率

        # ASPP (Neck) 增强多尺度上下文
        self.aspp = ASPP(in_channels=high_level_channels,
                         out_channels=aspp_out_channels,
                         atrous_rates=[6, 12, 18])

        # Decoder 融合低层细节
        self.decoder = Decoder(num_classes,
                               low_level_channels=low_level_channels,
                               aspp_out_channels=aspp_out_channels)

    def forward(self, x):
        # 记录原始输入尺寸，用于最后恢复分辨率
        input_size = x.shape[2:]

        # 提取特征
        features = self.backbone(x)
        # for name, feature in features.items():
        #     print(f'{name}: {feature.shape}')
        low_level_features = features['block2']   # 1/4 分辨率
        high_level_features = features['block4']  # 1/16 分辨率

        # ASPP (Neck) 增强多尺度上下文
        aspp_output = self.aspp(high_level_features)

        # Decoder 融合低层细节
        output = self.decoder(aspp_output, low_level_features)

        # 最后统一上采样，恢复到原始输入图像的分辨率 (1/1)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F



class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 模块

    五个并行分支：
      1x1 卷积
      3x3 空洞卷积 (rate=6)
      3x3 空洞卷积 (rate=12)
      3x3 空洞卷积 (rate=18)
      全局平均池化 (Image-level features)

    拼接后经 1x1 卷积融合输出。
    """
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()

        # 1x1 卷积分支
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 空洞卷积分支 (3x3, 不同 dilation rate)
        self.conv_atrous = nn.ModuleList()
        for rate in atrous_rates:
            self.conv_atrous.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 全局平均池化分支 (Image-level features)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 融合投影层: (1 + len(atrous) + 1) 个分支拼接
        num_branches = 1 + len(atrous_rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        features = []

        # 1x1 卷积分支
        features.append(self.conv_1x1(x))

        # 空洞卷积分支
        for conv in self.conv_atrous:
            features.append(conv(x))

        # 全局平均池化分支 → 上采样回原始空间尺寸
        pool_feat = self.global_pool(x)
        pool_feat = F.interpolate(pool_feat, size=size,
                                  mode='bilinear', align_corners=True)
        features.append(pool_feat)

        return self.project(torch.cat(features, dim=1))

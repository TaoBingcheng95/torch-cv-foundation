"""
AlexNet — 深度学习视觉革命的起点 (Krizhevsky et al., 2012)
============================================================

2012 年 ImageNet 图像分类竞赛冠军，以显著优势击败传统方法，
首次证明了深度学习在大规模视觉任务上的巨大潜力，直接引爆了
2012 年以后的深度学习浪潮。

相比 LeNet-5 的核心突破：
  1. 引入 ReLU 激活函数：替代 Sigmoid/Tanh，解决梯度消失问题，加速收敛
  2. 使用 Dropout：在全连接层随机丢弃神经元（p=0.2），有效抑制过拟合
  3. 更大的卷积核与更深的网络：5 个卷积层 + 3 个全连接层，参数量约 6000 万
  4. 数据增强：随机裁剪、水平翻转、颜色抖动，扩充训练数据多样性

原始论文使用两块 GPU 并行训练，本实现为单 GPU 版本。
输入尺寸: 3×224×224 (ImageNet 标准)
输出: 1000 类分类结果

参考论文: "ImageNet Classification with Deep Convolutional Neural Networks"
"""

import torch
import torch.nn as nn



class AlexNet(nn.Module):
    """
    AlexNet 的 PyTorch 实现。

    网络结构概览（以 3×224×224 输入为例）：
        特征提取: 5 个卷积层 (C1-C5) + 3 个最大池化层
        分类器:   3 个全连接层 (FC6-FC8)，前两层带 Dropout

    Args:
        in_channels: 输入图像通道数，RGB 图像为 3
        num_classes: 分类类别数，ImageNet 为 1000
        dropout:     全连接层的 Dropout 概率，默认 0.2
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 dropout: float = 0.2) -> None:
        super().__init__()

        # ============================================================
        # 特征提取部分：5 个卷积层 + 3 个最大池化层
        # ============================================================

        self.features = nn.Sequential(
            # C1: 3→64, 11×11 大卷积核, stride=4 快速降采样
            #     224×224 → 55×55  (公式: (224+2×2-11)/4+1 = 55)
            #     后接 3×3 最大池化, stride=2 → 27×27
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # C2: 64→192, 5×5 卷积核, padding=2 保持尺寸
            #     27×27 → 27×27  (公式: (27+2×2-5)/1+1 = 27)
            #     后接 3×3 最大池化, stride=2 → 13×13
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # C3: 192→384, 3×3 卷积核, padding=1 保持尺寸
            #     13×13 → 13×13  (无池化，连续卷积提取更丰富特征)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # C4: 384→256, 3×3 卷积核
            #     13×13 → 13×13  (无池化)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # C5: 256→256, 3×3 卷积核
            #     13×13 → 13×13
            #     后接 3×3 最大池化, stride=2 → 6×6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 自适应平均池化：将任意尺寸的特征图统一缩放到 6×6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # ============================================================
        # 分类器部分：3 个全连接层
        # FC6: 256×6×6 = 9216 → 4096  (Dropout + ReLU)
        # FC7: 4096 → 4096            (Dropout + ReLU)
        # FC8: 4096 → num_classes     (输出层，无激活)
        # ============================================================

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，张量维度变化如下（以 3×224×224 输入为例）：

        输入:  (3, 224, 224)
        C1+Pool: (64, 27, 27)   — 11×11 大卷积 + 池化，快速降采样
        C2+Pool: (192, 13, 13)  — 5×5 卷积 + 池化
        C3:      (384, 13, 13)  — 3×3 卷积，无池化
        C4:      (256, 13, 13)  — 3×3 卷积，无池化
        C5+Pool: (256, 6, 6)    — 3×3 卷积 + 池化
        AvgPool: (256, 6, 6)    — 自适应池化（此处尺寸已匹配，不改变）
        FC6:     (4096,)         — 展平 + 全连接 + Dropout
        FC7:     (4096,)
        Out:     (num_classes,)
        """
        # ---- 特征提取 ----
        x = self.features(x)        # (B, 3, 224, 224) → (B, 256, 6, 6)
        x = self.avgpool(x)         # (B, 256, 6, 6) — 确保空间维度为 6×6
        x = torch.flatten(x, 1)     # (B, 256, 6, 6) → (B, 9216)

        # ---- 分类器 ----
        x = self.classifier(x)      # (B, 9216) → (B, num_classes)
        return x

    def _initialize_weights(self):
        """
        权重初始化策略：
          - 卷积层: Kaiming 初始化（何教授方法），适配 ReLU 激活的方差特性
          - 全连接层: 小标准差正态分布 (std=0.01)
          - 偏置项: 全部初始化为 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



if __name__ == "__main__":
    from torchinfo import summary
    # from torchvision.models import AlexNet_Weights

    model = AlexNet(num_classes=1000)
    # model.load_state_dict(AlexNet_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True))

    input_size = (1, 3, 224, 224)
    summary(model, input_size=input_size)

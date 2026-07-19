"""
LeNet-5 — CNN 的开山之作 (LeCun et al., 1998)
==============================================

原始 LeNet-5 设计用于手写数字识别，接受 1×32×32 的灰度图像，输出 10 个类别。
本实现针对 MNIST 数据集（28×28）做了适配：
  - 在 C1 层使用 padding=2，将 28×28 补零到等效 32×32 的效果
  - 添加 AdaptiveMaxPool2d，确保进入全连接层前特征图尺寸严格为 5×5

原始网络结构（7 层）：
  [C1 卷积] → [S2 池化] → [C3 卷积] → [S4 池化] → [C5 卷积] → [F6 全连接] → [F7 全连接]
本实现中 C5 使用 nn.Linear 实现（等价于对 1×1 特征图做全连接），
并加入了 BatchNorm 以加速训练收敛（原始论文未使用）。

参考论文: "Gradient-Based Learning Applied to Document Recognition" (LeCun, 1998)
"""

import torch
import torch.nn as nn



class LeNet5(nn.Module):
    """
    LeNet-5 的 PyTorch 实现。

    网络结构概览（以 1×28×28 输入为例）：
        特征提取: Conv(1→6) → Pool → Conv(6→16) → Pool → AdaptivePool(→5×5)
        分类器:   FC(400→120) → FC(120→84) → FC(84→10)

    Args:
        num_classes: 分类类别数，MNIST 默认为 10（数字 0-9）
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ============================================================
        # 特征提取部分：两组 [卷积 + BN + ReLU + 池化]
        # ============================================================

        # 第 1 组 [C1 + S2]：
        #   Conv2d:  1 通道 → 6 通道, 5×5 卷积核, padding=2 补偿 28→32 的差异
        #            输出尺寸: 28×28 → 28×28 (padding=2 保持尺寸不变)
        #   BN:      对 6 个通道做批归一化，加速训练
        #   ReLU:    非线性激活
        #   MaxPool: 2×2 下采样, 28×28 → 14×14
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第 2 组 [C3 + S4]：
        #   Conv2d:  6 通道 → 16 通道, 5×5 卷积核, 无 padding
        #            输出尺寸: 14×14 → 10×10 (每边缩小 2 像素)
        #   MaxPool: 2×2 下采样, 10×10 → 5×5
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自适应池化：无论输入空间尺寸是多少，都强制输出 5×5
        # 这是为了兼容 MNIST 的 28×28 输入（原始 LeNet 设计为 32×32）
        self.adaptive_pool = nn.AdaptiveMaxPool2d((5, 5))

        # ============================================================
        # 分类器部分：三层全连接网络
        # 原始论文中 C5 是卷积层（对 5×5 特征图做 120 个 5×5 卷积），
        # 等价于将 16×5×5 展平后做全连接，这里直接用 Linear 实现。
        # ============================================================

        # C5&F6: 16×5×5 = 400 → 120
        self.fc = nn.Linear(16 * 5 * 5, 120)
        self.relu = nn.ReLU(inplace=True)

        # F7: 120 → 84
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU(inplace=True)

        # 输出层: 84 → num_classes (默认 10)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，张量维度变化如下（以 batch_size=1, 输入 1×28×28 为例）：

        输入:  (1, 28, 28)    — 单通道灰度图
        C1:    (6, 28, 28)    — 6 个 5×5 卷积核提取低级特征
        S2:    (6, 14, 14)    — 2×2 最大池化下采样
        C3:    (16, 10, 10)   — 16 个卷积核提取更复杂特征
        S4:    (16, 5, 5)     — 2×2 最大池化下采样
        Pool:  (16, 5, 5)     — 自适应池化保证尺寸
        F6:    (120,)          — 展平后全连接
        F7:    (84,)
        Out:   (num_classes,)
        """
        # ---- 特征提取 ----
        out = self.layer1(x)       # (B, 1, 28, 28) → (B, 6, 14, 14)
        out = self.layer2(out)     # (B, 6, 14, 14) → (B, 16, 5, 5)
        out = self.adaptive_pool(out)  # (B, 16, 5, 5) — 确保空间尺寸为 5×5

        # ---- 分类器 ----
        out = torch.flatten(out, 1)    # (B, 16, 5, 5) → (B, 400)
        out = self.relu(self.fc(out))  # (B, 400) → (B, 120)
        out = self.relu1(self.fc1(out))  # (B, 120) → (B, 84)
        out = self.fc2(out)            # (B, 84) → (B, num_classes)
        return out



if __name__ == "__main__":
    from torchinfo import summary
    
    model = LeNet5(num_classes=10)
    input_size = (1, 1, 28, 28)
    summary(model, input_size=input_size)

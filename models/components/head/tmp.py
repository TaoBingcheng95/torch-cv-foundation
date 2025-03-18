import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=256):
        super().__init__()
        self.num_classes = num_classes

        # 上采样分支（可根据需要调整上采样方式）
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, 2, 1),  # 转置卷积上采样
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 细化分支
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1)  # 最终分类层
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.refine(x)
        return x


if __name__ == '__main__':
    # 输入特征图的通道数为 2048，目标输出分辨率为 224x224，类别数为 21
    seg_head = SegmentationHead(in_channels=2048, num_classes=21)
    # 假设输入特征图的分辨率为 7x7
    input_tensor = torch.randn(1, 2048, 7, 7)
    input_resolution = (7, 7)

    # 前向传播
    output = seg_head(input_tensor)
    print("Segmentation output shape:", output.shape)  # 输出形状: (1, 21, 224, 224)

    # # 使用示例 ---------------------------------------------------
    # # 假设骨干网络输出特征图尺寸为 [batch, 512, H/16, W/16]
    # backbone_out = torch.randn(4, 512, 32, 32)  # 示例输入
    #
    # # 初始化任务头（假设21个类别）
    # seg_head = SegmentationHead(in_channels=512, num_classes=21)
    #
    # # 前向传播
    # output = seg_head(backbone_out)  # 输出形状 [4, 21, 64, 64]
    # print(output.shape)
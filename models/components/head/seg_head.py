import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, target_resolution=(224, 224)):
        """
        初始化通用分割头

        :param in_channels: 输入特征图的通道数
        :param num_classes: 分割任务的类别数
        :param target_resolution: 目标输出分辨率 (H, W)
        """
        super(GenericSegmentationHead, self).__init__()
        self.target_resolution = target_resolution

        # 上采样倍数和层数
        self.upsample_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # 初始通道数
        current_channels = in_channels

        # 动态计算需要的上采样层数
        while True:
            # 添加上采样层
            self.upsample_layers.append(
                nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=4, stride=2, padding=1)
            )
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels // 2, current_channels // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = current_channels // 2

            # 检查是否达到目标分辨率
            if current_channels < 128:  # 最小通道数 32
                break
        # 最后的卷积层，输入通道数为 current_channels，输出通道数为 num_classes
        self.final_conv = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(self, x, input_resolution):
        """
        前向传播

        :param x: 输入特征图
        :param input_resolution: 输入特征图的分辨率 (H, W)
        :return: 分割结果（每个像素的类别得分）
        """
        # 动态计算需要的上采样倍数
        target_H, target_W = self.target_resolution
        current_H, current_W = input_resolution

        # 上采样到目标分辨率
        for upsample_layer, conv_layer in zip(self.upsample_layers, self.conv_layers):
            x = upsample_layer(x)
            x = conv_layer(x)
            current_H *= 2
            current_W *= 2

            # 如果已经达到目标分辨率，提前退出
            if current_H >= target_H and current_W >= target_W:
                break

        # 如果未达到目标分辨率，使用插值上采样
        if current_H != target_H or current_W != target_W:
            x = F.interpolate(x, size=self.target_resolution, mode="bilinear", align_corners=False)

        # 最后的卷积层
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    # 输入特征图的通道数为 2048，目标输出分辨率为 224x224，类别数为 21
    seg_head = GenericSegmentationHead(in_channels=2048, num_classes=21, target_resolution=(224, 224))
    # 假设输入特征图的分辨率为 7x7
    input_tensor = torch.randn(1, 2048, 7, 7)
    input_resolution = (7, 7)

    # 前向传播
    output = seg_head(input_tensor, input_resolution)
    print("Segmentation output shape:", output.shape)  # 输出形状: (1, 21, 224, 224)

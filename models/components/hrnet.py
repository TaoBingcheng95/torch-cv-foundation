# https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class HRNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(HRNetStage, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            self.blocks.add_module(f'block{i}', BasicBlock(in_channels, out_channels, stride, downsample))
            in_channels = out_channels

    def forward(self, x):
        return self.blocks(x)


class HRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRNet, self).__init__()
        self.preconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.prebn = nn.BatchNorm2d(64)
        self.prerelu = nn.ReLU(inplace=True)

        # HRNet stages
        self.stage1 = HRNetStage(64, 64, 4)
        self.stage2 = HRNetStage(64, 128, 4)
        self.stage3 = HRNetStage(128, 256, 4)
        self.stage4 = HRNetStage(256, 512, 4)

        # Final convolutional layer for classification
        self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        shape = (x.shape[2], x.shape[3])
        x = self.preconv(x)
        x = self.prebn(x)
        x = self.prerelu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)

        return x


# 示例测试
if __name__ == "__main__":
    model = HRNet(in_channels=3, out_channels=21)
    x = torch.randn(1, 3, 256, 256)  # 输入为 2 张 256x256 的图像
    output = model(x)
    print(output.shape)  # 输出尺寸应为 (2, 21, 1024, 1024)

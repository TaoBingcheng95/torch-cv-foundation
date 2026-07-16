import os
import torch
import torch.nn as nn
import torch.nn.functional as F   # 新增导入
# import torchvision.models as models

from .backbone import VGG16Encoder, ResNet18Encoder, MobileNetV2Encoder



class DoubleConv(nn.Module):
    """
    核心卷积块：(3x3 Conv => ReLU) * 2
    这是 U-Net 中 Encoder 和 Decoder 的基本组成单元。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 注意：这里使用了 padding=1，这是现代实现支持任意尺寸输入的关键！
        # 原论文没有 padding，会导致特征图不断缩小。
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class EncoderBlock(nn.Module):
    """
    ContractBlock
    下采样模块 (Encoder 的一部分)：
    先使用 DoubleConv 提取特征并翻倍通道数， 然后进行 2x2 MaxPool 将尺寸减半，。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p



class DecoderBlock(nn.Module):
    """
    ExpandBlock
    上采样模块 (Decoder 的一部分)：
    先上采样将尺寸翻倍、通道减半，然后与 Encoder 的对应层拼接 (Skip Connection)，
    最后通过 DoubleConv 融合特征（自动尺寸对齐）。
    支持任意通道数的 Skip Connection
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 上采样（不改变通道数）
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 【新增】通道对齐层：将深层特征强制转换为 skip 特征的通道数
        self.reduce = nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False)
        
        # 拼接后，总通道数严格等于 skip_channels * 2
        self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce(x1) # 关键：将 x1 的通道数对齐到 x2 (skip)
        
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)



class UNet(nn.Module):
    """标准UNet（Ronneberger et al., 2015），支持任意输入尺寸（建议偶数）"""
    def __init__(self, in_channels=3, num_classes=1, features=64):
        super().__init__()

        # 第一层没有池化，只有双卷积
        self.inc = DoubleConv(in_channels, 64)

        # 编码器
        self.down1 = EncoderBlock(features, features * 2) # 64, 128
        self.down2 = EncoderBlock(features*2, features * 4) # 128, 256
        self.down3 = EncoderBlock(features * 4, features * 8) # 256,512
        self.down4 = EncoderBlock(features * 8, features * 16) # 512, 1024
  
        # 解码器
        self.up1 = DecoderBlock(features * 16, features * 8) # 1024, 512
        self.up2 = DecoderBlock(features * 8, features * 4) # 512,256
        self.up3 = DecoderBlock(features * 4, features * 2) # 256,128
        self.up4 = DecoderBlock(features * 2, features) # 128,64

        # 输出层
        self.out = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):
        # 基础特征提取
        x_init = self.inc(x)

        # Encoder 阶段，返回 skip 特征 和 下采样结果
        skip1, x_down1 = self.down1(x_init)
        skip2, x_down2 = self.down2(x_down1)
        skip3, x_down3 = self.down3(x_down2)
        skip4, x_down4 = self.down4(x_down3) # x_down4 是 Bottleneck
        
        # # Decoder 阶段：将 skip 特征与深层特征融合
        x_up4 = self.up1(x_down4, skip3)
        x_up3 = self.up2(x_up4, skip2)
        x_up2 = self.up3(x_up3, skip1)
        x_up1 = self.up4(x_up2, x_init)

        logits = self.out(x_up1)

        return logits



class UNet_ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained_encoder=False):
        super().__init__()
        
        # Encoder: 5 阶段，32 倍下采样
        self.encoder = ResNet18Encoder(pretrained=pretrained_encoder)
        
        # 通道数对齐层：将 initial 的输出 (64通道) 降维到 32通道，
        # 以便与上一层 Decoder 输出的 32 通道完美拼接 (32+32=64)
        self.align_x0 = nn.Conv2d(64, 32, kernel_size=1)
        
        # Decoder (通道数严格匹配 ResNet18 的输出)
        # up4: 接收 512 -> up变256 + skip3(256) = 512 -> 输出 256
        self.up4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # up3: 接收 256 -> up变128 + skip2(128) = 256 -> 输出 128
        self.up3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # up2: 接收 128 -> up变64  + skip1(64)  = 128 -> 输出 64
        self.up2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # up1: 接收 64  -> up变32  + align_x0(32)= 64  -> 输出 32
        self.up1 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)

        # 最终上采样层：将 1/2 尺寸恢复到 1（原图尺寸）
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 4. 输出层
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        
        #  Encoder: 5 阶段，32 倍下采样
        features = self.encoder(x)
        x0, x1, x2, x3, x4 = features

        # # 对齐最浅层特征的通道数
        x0_aligned = self.align_x0(x0)
        
        # Decoder 阶段：自底向上融合
        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0_aligned)

        # 最终上采样恢复原图尺寸
        d0 = self.final_up(d1)
        
        # 输出
        out = self.out(d0)
        return out



class UNet_MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained_encoder=False):
        super().__init__()
        
        # 1. Encoder: 通道数序列 [16, 24, 32, 96, 320]
        self.encoder = MobileNetV2Encoder(pretrained=pretrained_encoder)
        
        # 2. Decoder: 严格对齐通道数
        # up4: 接收 f4(320), skip 是 f3(96). 拼接后 96*2=192. 输出 96
        self.up4 = DecoderBlock(in_channels=320, skip_channels=96, out_channels=96)
        
        # up3: 接收 96, skip 是 f2(32). 拼接后 32*2=64. 输出 32
        self.up3 = DecoderBlock(in_channels=96, skip_channels=32, out_channels=32)
        
        # up2: 接收 32, skip 是 f1(24). 拼接后 24*2=48. 输出 24
        self.up2 = DecoderBlock(in_channels=32, skip_channels=24, out_channels=24)
        
        # up1: 接收 24, skip 是 f0(16). 拼接后 16*2=32. 输出 16
        self.up1 = DecoderBlock(in_channels=24, skip_channels=16, out_channels=16)

        # 3. 最终上采样：将 1/2 尺寸恢复为原图尺寸
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 4. 输出层
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)
        
        d4 = self.up4(f4, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)
        
        d0 = self.final_up(d1)
        return self.out(d0)




if __name__ == "__main__":
    model = UNet_MobileNetV2(in_channels=3, num_classes=1)
    x = torch.randn(1, 3, 244, 244)  # 原论文输入尺寸572x572
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

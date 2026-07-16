
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: int = 64):
        """
        Args:
            in_channels: 输入图像的通道数（RGB为3，灰度图为1）
            out_channels: 输出分割图的通道数（二分类为1，多分类为类别数）
            features: 第一层卷积的滤波器数量，按原论文设为64
        """
        super().__init__()

        # ---------- 编码器 (Contracting Path) ----------
        # 每一层：DoubleConv -> 下采样（MaxPool2d）
        self.enc1 = DoubleConv(in_channels, features)          # 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(features, features * 2)         # 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(features * 2, features * 4)     # 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(features * 4, features * 8)     # 512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- 瓶颈层 (Bottleneck) ----------
        # 原论文最底层为 1024 个通道
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # ---------- 解码器 (Expansive Path) ----------
        # 每一步：上采样 -> 拼接（skip connection）-> DoubleConv
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features * 16, features * 8)    # 拼接后通道数翻倍：1024 -> 512

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4)     # 512 -> 256

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)     # 256 -> 128

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features * 2, features)         # 128 -> 64

        # ---------- 输出层 ----------
        # 1x1 卷积将特征图映射到目标类别数
        self.out = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # ---------- 编码器 ----------
        enc1 = self.enc1(x)           # 保存用于跳跃连接
        p1 = self.pool1(enc1)

        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)

        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)

        enc4 = self.enc4(p3)
        p4 = self.pool4(enc4)

        # ---------- 瓶颈 ----------
        bottleneck = self.bottleneck(p4)

        # ---------- 解码器（含跳跃连接）----------
        # 上采样 -> 与对应编码器输出拼接 -> DoubleConv
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat((d4, enc4), dim=1)      # 沿通道维度拼接[reference:7]
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, enc3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, enc2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, enc1), dim=1)
        d1 = self.dec1(d1)

        # ---------- 输出 ----------
        out = self.out(d1)
        return out



# UNet
class BaseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()
        
        # Encoder path (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder path (upsampling)
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        # Final convolution layer to get desired output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        return self.final_conv(dec1)





if __name__ == '__main__':
    from torchinfo import summary


    input_size = (1,3,576, 576)
    x = torch.randn(input_size)
    model = BaseUNet(in_channels =3, out_channels=10)
    y= model(x)
    # print(y.shape)
    # summary(model, input_size)

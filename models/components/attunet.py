# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 0:09
# @Author  : xuxing
# @Site    : 
# @File    : attunet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


# AttentionBlock
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, g_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(g_channels, in_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.W_h(psi)
        psi = torch.sigmoid(self.psi(psi))
        return x * psi


# UNet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(UNet, self).__init__()
        self.use_attention = use_attention
        
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
        
        # Attention blocks
        if self.use_attention:
            self.attention4 = AttentionBlock(512, 512)
            self.attention3 = AttentionBlock(256, 256)
            self.attention2 = AttentionBlock(128, 128)
            self.attention1 = AttentionBlock(64, 64)
    
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
        if self.use_attention:
            att4 = self.attention4(dec4, enc4)
            dec4 = torch.cat((dec4, att4), dim=1)
        else:
            dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        if self.use_attention:
            att3 = self.attention3(dec3, enc3)
            dec3 = torch.cat((dec3, att3), dim=1)
        else:
            dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if self.use_attention:
            att2 = self.attention2(dec2, enc2)
            dec2 = torch.cat((dec2, att2), dim=1)
        else:
            dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if self.use_attention:
            att1 = self.attention1(dec1, enc1)
            dec1 = torch.cat((dec1, att1), dim=1)
        else:
            dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        return self.final_conv(dec1)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=2, use_attention=False)  # For a single-channel output (e.g., binary segmentation)
    # print(model)

    input_size = (1, 3, 512, 512)
    x = torch.randn(input_size)  # Example input tensor (batch_size, channels, height, width)
    # summary(model, input_size=(1, 3, 512, 512))
    output = model(x)
    print(output.shape)  # Should be (1, 1, 256, 256) for this example

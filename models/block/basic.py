import torch
from torch import nn


class BasicConvBlock(nn.Module):
    """
    Conv+BatchNorm+Relu
    """
    def __init__(self, in_channels, out_channels, k=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    """ BasicConvBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = BasicConvBlock(in_dim, out_dim)
        self.conv_2 = BasicConvBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class TripleConvBlock(nn.Module):
    """ BasicConvBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = BasicConvBlock(in_dim, out_dim)
        self.conv_2 = BasicConvBlock(out_dim, out_dim)
        self.conv_3 = BasicConvBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Simple fully convolutional neural network (FCN) implementations."""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torchvision import models


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    return a bilinear filter tensor
    双线性卷积核，用于反卷积
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCN(Module):
    """
    A simple 5 layer FCN with leaky relus and 'same' padding.
    """

    def __init__(self, in_channels: int=3,
                 classes: int=10,
                 num_filters: int = 64,
                 dropout_prob: float = 0.5,  # Dropout 概率
                 ) -> None:
        """
        Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
            dropout_prob: Dropout probability (default: 0.5)
        """
        super().__init__()

        # conv1 = nn.modules.Conv2d(
        #     in_channels, num_filters, kernel_size=3, stride=1, padding=1
        # )
        # conv2 = nn.modules.Conv2d(
        #     num_filters, num_filters, kernel_size=3, stride=1, padding=1
        # )
        # conv3 = nn.modules.Conv2d(
        #     num_filters, num_filters, kernel_size=3, stride=1, padding=1
        # )
        # conv4 = nn.modules.Conv2d(
        #     num_filters, num_filters, kernel_size=3, stride=1, padding=1
        # )
        # conv5 = nn.modules.Conv2d(
        #     num_filters, num_filters, kernel_size=3, stride=1, padding=1
        # )
        # self.backbone = nn.modules.Sequential(
        #     conv1,
        #     nn.modules.LeakyReLU(inplace=True),
        #     conv2,
        #     nn.modules.LeakyReLU(inplace=True),
        #     conv3,
        #     nn.modules.LeakyReLU(inplace=True),
        #     conv4,
        #     nn.modules.LeakyReLU(inplace=True),
        #     conv5,
        #     nn.modules.LeakyReLU(inplace=True),
        # )
        # 定义卷积层 + 批归一化 + 激活函数 + Dropout
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),  # 批归一化
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_prob),  # Dropout

            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_prob),

            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_prob),

            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_prob),

            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

        # 最后的 1x1 卷积层
        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

        # 分类任务头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将空间维度压缩为 1x1
            nn.Flatten(),  # 将 4D 张量展平为 2D 张量 (batch_size, num_filters)
            nn.Linear(num_filters, classes),  # 全连接层，输出类别数
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.classifier(x)
        # x = self.last(x)
        return x


class FCN32s(Module):
    def __init__(self, num_classes=5):
        super(FCN32s, self).__init__()
        self.pretrained_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature = self.pretrained_model.features

        self.conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.upsample32x = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes,
                               kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes,
                               kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes,
                               kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes,
                               kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes,
                               kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_kernel(int(m.in_channels), int(m.out_channels), m.kernel_size[0]))

    def forward(self, x):
        x = self.feature(x)  # 1/32
        x = self.conv(x)
        x = self.upsample32x(x)
        return x


class FCN16s(nn.Module):
    def __init__(self,num_classes):
        super(FCN16s, self).__init__()
        self.pretrained_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_1=nn.Sequential(*list(self.pretrained_model.features.children())[:24])
        self.feature_2=nn.Sequential(*list(self.pretrained_model.features.children())[24:])

        self.conv_1=nn.Conv2d(512,num_classes,kernel_size=1,stride=1,padding=0)
        self.conv_2=nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

        self.upsample2x=nn.ConvTranspose2d(num_classes,num_classes,kernel_size=3,stride=2,padding=1,output_padding=1,dilation=1)
        self.upsample16x=nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        )

        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_kernel(m.in_channels,m.out_channels,m.kernel_size[0]))

    def forward(self, x):
        x1=self.feature_1(x)
        x2=self.feature_2(x1)

        x1=self.conv_1(x1)
        x2=self.conv_2(x2)
        x2=self.upsample2x(x2)
        x2+=x1

        x2=self.upsample16x(x2)
        return x2


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.pretrained_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_1 = nn.Sequential(*list(self.pretrained_model.features.children())[:17])
        self.feature_2 = nn.Sequential(*list(self.pretrained_model.features.children())[17:24])
        self.feature_3 = nn.Sequential(*list(self.pretrained_model.features.children())[24:])

        self.conv_1 = nn.Conv2d(512,num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_2=nn.Conv2d(256,num_classes,kernel_size=1, stride=1, padding=0)
        self.conv_3=nn.Conv2d(512,num_classes, kernel_size=1, stride=1, padding=0)

        self.upsample2x_1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.upsample2x_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1,dilation=1)
        self.upsample8x = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data=bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])

    def forward(self, x):
        x1 = self.feature_1(x)
        x2 = self.feature_2(x1)
        x3 = self.feature_3(x2)

        x2 = self.conv_1(x2)
        x3 = self.conv_3(x3)
        x3 = self.upsample2x_1(x3)
        x3 += x2

        x1 = self.conv_2(x1)
        x3 = self.upsample2x_2(x3)
        x3 += x1

        x3 = self.upsample8x(x3)
        return x3



if __name__ == '__main__':
    from torchinfo import summary

    input_size = (1, 3, 512, 512)
    # input_data = torch.randn(input_size)

    #model = FCN(in_channels=3, classes=5)
    model = FCN32s()
    # print(model)
    # output = model(input_data)
    # print(output.shape)
    summary(model,
            input_size=input_size,
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings=['var_names'],
            verbose=True
            )

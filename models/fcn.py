
import os
import numpy as np

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights



def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """生成双线性插值的转置卷积权重核"""
    weight = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    y = torch.arange(kernel_size, dtype=torch.float32)
    x = torch.arange(kernel_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    # 计算双线性核权重
    kernel_2d = (1 - torch.abs(xx - center) / factor) * (1 - torch.abs(yy - center) / factor)
    # 赋值给对角线通道
    for i in range(in_channels):
        weight[i, i] = kernel_2d
    return weight



class FCN32s(nn.Module):
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # # conv1
        # self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        # self.relu1_1 = nn.ReLU(inplace=True)
        # self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.relu1_2 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        # # conv2
        # self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.relu2_1 = nn.ReLU(inplace=True)
        # self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.relu2_2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        # # conv3
        # self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        # self.relu3_1 = nn.ReLU(inplace=True)
        # self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.relu3_2 = nn.ReLU(inplace=True)
        # self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.relu3_3 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        # # conv4
        # self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        # self.relu4_1 = nn.ReLU(inplace=True)
        # self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu4_2 = nn.ReLU(inplace=True)
        # self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu4_3 = nn.ReLU(inplace=True)
        # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        # # conv5
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_2 = nn.ReLU(inplace=True)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_3 = nn.ReLU(inplace=True)
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone=pretrained_model.features

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        # score 层
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # self.fcn_head = nn.Sequential(
        #     self.fc6,
        #     self.relu6,
        #     self.drop6,
        #     self.fc7,
        #     self.relu7,
        #     self.drop7,
        #     self.score_fr,
        # )

        self.upsample32 = nn.ConvTranspose2d(num_classes, num_classes, 
                                             kernel_size=64, stride=32, padding=16, bias=False)

        self._initialize_weights(pretrained_model=pretrained_model)


    def forward(self, x):
        input_size = x.size()[2:]
        pool5 = self.backbone(x)

        # h = self.relu6(self.fc6(pool5))
        # h = self.drop6(h)
        # h = self.relu7(self.fc7(h))
        # h = self.drop7(h)
        # h = self.score_fr(h)
        x = self.fcn_head(pool5)

        out = self.upsample32(x)
        # or 直接使用 F.interpolate 上采样 32 倍
        # out = F.interpolate(h, size=input_size, mode='bilinear', align_corners=False)

        return out

    def _initialize_weights(self, pretrained_model):
        # 1. 从 VGG16 的 classifier 中迁移 fc6 和 fc7 的权重
        vgg_classifier = pretrained_model.classifier
        
        # 迁移 fc6 (对应 classifier[0])
        fc6_weight = vgg_classifier[0].weight.data # shape: (4096, 25088)
        fc6_bias = vgg_classifier[0].bias.data
        self.fc6.weight.data = fc6_weight.view(4096, 512, 7, 7)
        self.fc6.bias.data = fc6_bias

        # 迁移 fc7 (对应 classifier[3])
        fc7_weight = vgg_classifier[3].weight.data # shape: (4096, 4096)
        fc7_bias = vgg_classifier[3].bias.data
        self.fc7.weight.data = fc7_weight.view(4096, 4096, 1, 1)
        self.fc7.bias.data = fc7_bias

        # 2. 初始化 score_fr (论文推荐：均值为0，标准差为0.01)
        nn.init.normal_(self.score_fr.weight, mean=0, std=0.01)
        nn.init.constant_(self.score_fr.bias, 0)

        # 3. 初始化 upsample32 为双线性插值核
        initial_weight = get_upsampling_weight(
            self.upsample32.in_channels, 
            self.upsample32.out_channels, 
            self.upsample32.kernel_size[0]
        )
        self.upsample32.weight.data.copy_(initial_weight)
        
        # 冻结 backbone 的前几层（可选，视具体训练策略而定）
        # for param in self.backbone[:10].parameters():
        #     param.requires_grad = False



class FCN16s(nn.Module):
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone=pretrained_model.features
        self.feature_1=nn.Sequential(*list(self.backbone.children())[:24])
        self.feature_2=nn.Sequential(*list(self.backbone.children())[24:])

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        # score 层
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
  
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes,
                                            kernel_size = 4,stride = 2, padding = 1,
                                            bias = False)
        self.upsample16 = nn.ConvTranspose2d(num_classes, num_classes, 
                                             kernel_size=32, stride=16, padding=8,
                                             bias=False)

        self._initialize_weights(pretrained_model)


    def forward(self, x):
        # pool5 = self.backbone(x)
        pool4 = self.feature_1(x)
        pool5 = self.feature_2(pool4)

        h = self.relu6(self.fc6(pool5))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upsample2(h)

        pool4_score = self.score_pool4(pool4)
        h = h + pool4_score

        out = self.upsample16(h)

        return out


    def _initialize_weights(self, pretrained_model):
        # 1. 从 VGG16 的 classifier 中迁移 fc6 和 fc7 的权重
        vgg_classifier = pretrained_model.classifier
        
        # 迁移 fc6 (对应 classifier[0])
        fc6_weight = vgg_classifier[0].weight.data # shape: (4096, 25088)
        fc6_bias = vgg_classifier[0].bias.data
        self.fc6.weight.data = fc6_weight.view(4096, 512, 7, 7)
        self.fc6.bias.data = fc6_bias

        # 迁移 fc7 (对应 classifier[3])
        fc7_weight = vgg_classifier[3].weight.data # shape: (4096, 4096)
        fc7_bias = vgg_classifier[3].bias.data
        self.fc7.weight.data = fc7_weight.view(4096, 4096, 1, 1)
        self.fc7.bias.data = fc7_bias

        # 2. 初始化 score_fr (论文推荐：均值为0，标准差为0.01)
        nn.init.normal_(self.score_fr.weight, mean=0, std=0.01)
        nn.init.constant_(self.score_fr.bias, 0)
        nn.init.normal_(self.score_pool4.weight, mean=0, std=0.01)
        nn.init.constant_(self.score_pool4.bias, 0)

        # 3. 初始化所有的上采样层为双线性插值核 (补充 upsample2)
        # upsample2 (2倍上采样)
        initial_weight_2 = get_upsampling_weight(
            self.upsample2.in_channels, 
            self.upsample2.out_channels, 
            self.upsample2.kernel_size[0]
        )
        self.upsample2.weight.data.copy_(initial_weight_2)

        # upsample16 (16倍上采样)
        initial_weight_16 = get_upsampling_weight(
            self.upsample16.in_channels, 
            self.upsample16.out_channels, 
            self.upsample16.kernel_size[0]
        )
        self.upsample16.weight.data.copy_(initial_weight_16)
        
        # 冻结 backbone 的前几层（可选，视具体训练策略而定）
        # for param in self.backbone[:10].parameters():
        #     param.requires_grad = False



class FCN8s(nn.Module):
    def __init__(self, num_classes: int =10):
        super().__init__()
        pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone=pretrained_model.features
        self.backbone_1 = nn.Sequential(*list(self.backbone.children())[:17])
        self.backbone_2 = nn.Sequential(*list(self.backbone.children())[17:24])
        self.backbone_3 = nn.Sequential(*list(self.backbone.children())[24:])

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # score 层
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        
        self.upsample2_1 = nn.ConvTranspose2d(num_classes,num_classes,
                                              kernel_size = 4,stride = 2,padding = 1,bias = False)
        self.upsample2_2 = nn.ConvTranspose2d(num_classes, num_classes, 
                                              kernel_size=4, stride=2, padding=1,bias=False)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=16, stride=8, padding=4,bias=False)
        

        self._initialize_weights(pretrained_model)


    def forward(self, x):
   
        pool3 = self.backbone_1(x)     # maxpooling3的feature map (1/8)
        pool4 = self.backbone_2(pool3) # maxpooling4的feature map (1/16)
        pool5 = self.backbone_3(pool4) # maxpooling5的feature map (1/32)

        h = self.relu6(self.fc6(pool5))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        upscore2 = self.upsample2_1(h) # 1/16
        pool4_score = self.score_pool4(pool4) # 1/16
        h = upscore2 + pool4_score  # 1/16

        upscore_pool4 = self.upsample2_2(h) # 1/8
        pool3_score = self.score_pool3(pool3) # 1/8
        h = upscore_pool4 + pool3_score  # 1/8

        h = self.upsample8(h)

        return h

    def _initialize_weights(self, pretrained_model):
        # 1. 从 VGG16 的 classifier 中迁移 fc6 和 fc7 的权重
        vgg_classifier = pretrained_model.classifier
        
        # 迁移 fc6 (对应 classifier[0])
        fc6_weight = vgg_classifier[0].weight.data # shape: (4096, 25088)
        fc6_bias = vgg_classifier[0].bias.data
        self.fc6.weight.data = fc6_weight.view(4096, 512, 7, 7)
        self.fc6.bias.data = fc6_bias

        # 迁移 fc7 (对应 classifier[3])
        fc7_weight = vgg_classifier[3].weight.data # shape: (4096, 4096)
        fc7_bias = vgg_classifier[3].bias.data
        self.fc7.weight.data = fc7_weight.view(4096, 4096, 1, 1)
        self.fc7.bias.data = fc7_bias

        # 2. 初始化 score_fr (论文推荐：均值为0，标准差为0.01)
        nn.init.normal_(self.score_fr.weight, mean=0, std=0.01)
        nn.init.constant_(self.score_fr.bias, 0)
        nn.init.normal_(self.score_pool4.weight, mean=0, std=0.01)
        nn.init.constant_(self.score_pool4.bias, 0)
        nn.init.normal_(self.score_pool3.weight, mean=0, std=0.01)
        nn.init.constant_(self.score_pool3.bias, 0)

        # 3. 初始化所有的上采样层为双线性插值核 (补充 upsample2)
        initial_weight_2 = get_upsampling_weight(
            self.upsample2_1.in_channels, 
            self.upsample2_1.out_channels, 
            self.upsample2_1.kernel_size[0]
        )
        # upsample2_1 (2倍上采样)
        self.upsample2_1.weight.data.copy_(initial_weight_2)
        # upsample2_2 (2倍上采样)
        self.upsample2_2.weight.data.copy_(initial_weight_2)

        # upsample8 (8倍上采样)
        initial_weight_8 = get_upsampling_weight(
            self.upsample8.in_channels, 
            self.upsample8.out_channels, 
            self.upsample8.kernel_size[0]
        )
        self.upsample8.weight.data.copy_(initial_weight_8)
        
        # 冻结 backbone 的前几层（可选，视具体训练策略而定）
        # for param in self.backbone[:10].parameters():
        #     param.requires_grad = False



class SimpleFCN(nn.Module):
    """
    A simple 5 layer FCN with leaky relus and 'same' padding, no Pooling layer.
    """

    def __init__(self, 
                 in_channels: int=3,
                 num_classes: int=10,
                 num_filters: int = 64,
                 dropout_prob: float = 0.1,  # Dropout 概率
                 ) -> None:
        """
        Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            num_classes: Number of output classes (channels in the final layer)
            num_filters: Number of filters in each convolutional layer/隐藏层卷积的通道数
            dropout_prob: Dropout probability (default: 0.5)
        
        FCN 的输出是“图”，而不是分类网络输出的“向量”。训练时对应的损失函数应该是 nn.CrossEntropyLoss（输入为 [B, C, H, W] 的 logits）
        """
        super().__init__()

        # 定义卷积层 + 批归一化 + 激活函数 + Dropout
        # 使用循环构建重复模块，避免代码冗余
        layers = []
        current_channels = in_channels
        for i in range(5):
            layers.extend([
                nn.Conv2d(current_channels, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.LeakyReLU(inplace=True),
            ])
            # 只在最后一层添加一个较小的 Dropout
            if i == 4: 
                layers.append(nn.Dropout2d(dropout_prob))
            current_channels = num_filters
        self.backbone = nn.Sequential(*layers)

        # 分割任务头
        # 1x1 卷积层：用于将特征通道数映射到类别数 (FCN 的核心分类头)
        self.last = nn.Conv2d(
            num_filters, num_classes, kernel_size=1, stride=1, padding=0
        )

        # # 分类任务头
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),              # 全局平均池化，将空间维度压缩为 1x1
        #     nn.Flatten(),                         # 将 4D 张量展平为 2D 张量 (batch_size, num_filters)
        #     nn.Linear(num_filters, num_classes),  # 全连接层，输出类别数
        # )

        # 权重初始化 (教学建议：给初学者展示基础的初始化方法)
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.
        Args:
            x: 输入张量，形状为 [Batch, in_channels, Height, Width]
        Returns:
            输出张量，形状为 [Batch, num_classes, Height, Width]
        """
        x = self.backbone(x)
        # out = self.classifier(x)
        out = self.last(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




if __name__ == '__main__':
    from torchinfo import summary

    input_size = [1, 3, 224, 224]
    input_data = torch.randn(1, 3, 224, 224)  # batch_size, channel，h, w
    
    model = SimpleFCN()

    # out = model(input_data)
    # print(out.shape)
    summary(model, input_size=input_size)


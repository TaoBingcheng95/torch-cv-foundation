
import torch
import torch.nn as nn
import timm


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.atrous_blocks = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
            )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_features = [block(x) for block in self.atrous_blocks]
        global_avg = self.global_avg_pool(x)
        global_avg = nn.functional.interpolate(global_avg, size=x.shape[2:], mode='bilinear', align_corners=False)
        atrous_features.append(global_avg)
        x = torch.cat(atrous_features, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, low_level_inplanes, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.last_conv = nn.Sequential(
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = nn.functional.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, out_channels, backbone='resnet18', output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        
        # 使用 timm 加载主干网络
        self.backbone = timm.create_model(backbone, pretrained=False, features_only=True)
        
        # 动态获取主干网络的输出通道数
        backbone_out_channels = self.backbone.feature_info.channels()  # 获取所有特征层的输出通道数
        
        # 选择主干网络的最后一个特征图作为 ASPP 的输入
        inplanes = backbone_out_channels[-1]
        low_level_inplanes = backbone_out_channels[0]  # 通常使用第一个特征图作为解码器的低级特征
        
        # 自适应通道转换
        self.channel_transform = nn.Conv2d(inplanes, 256, kernel_size=1)
        
        # ASPP模块
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise NotImplementedError("Output stride not supported")
        
        self.aspp = ASPP(256, 256, atrous_rates)
        
        # 解码器模块
        self.decoder = Decoder(low_level_inplanes, out_channels)
    
    def forward(self, x):
        shape = (x.shape[2], x.shape[3])
        # 从主干网络提取特征
        features = self.backbone(x)
        low_level_features = features[0]  # 低级特征
        x = features[-1]  # 高级特征
        
        # 通道转换
        x = self.channel_transform(x)
        
        # 通过 ASPP 和解码器
        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        x = nn.functional.interpolate(x, size=shape, mode='bilinear', align_corners=False)
        return x


if __name__ == '__main__':
    num_classes = 21
    model = DeepLabV3Plus(out_channels=num_classes)  # For a single-channel output (e.g., binary segmentation)
    # print(model)
    
    x = torch.randn((1, 3, 256, 256))  # Example input tensor (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)  # Should be (1, 1, 256, 256) for this example

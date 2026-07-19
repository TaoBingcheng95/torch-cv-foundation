"""
VGG — 小卷积核的胜利 (Simonyan & Zisserman, 2014)
===================================================

2014 年 ImageNet 图像分类竞赛亚军（同时获得定位竞赛冠军），
核心贡献：用连续的 3×3 小卷积核替代大卷积核（如 5×5、7×7），
在保持相同感受野的同时减少参数量、增加非线性。

相比前代网络的关键突破：
  1. 堆叠 3×3 卷积：两层 3×3 等价于一层 5×5 的感受野，但参数更少、非线性更强
  2. 统一的结构：所有卷积层均为 3×3、stride=1、padding=1，仅通过池化层降采样
  3. 更深的网络：VGG16 达 16 层，VGG19 达 19 层，远超 AlexNet 的 8 层
  4. 大规模参数：约 1.38 亿参数，其中全连接层占绝大多数

本文件提供两种实现方式：
  - VGG16 / VGG19：硬编码的经典结构，便于逐层阅读
  - VGG + cfgs：通过配置文件动态生成不同变体（VGG11/13/16/19），
    展示如何用配置驱动模型构建，是 torchvision 的官方实现风格

输入尺寸: 3×224×224 (ImageNet 标准)
输出: 1000 类分类结果

参考论文: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
"""

from typing import cast, Optional, Union

import torch
import torch.nn as nn
# import torch.functional as F
from torchvision.models import WeightsEnum

try:
    from .utils.pytorch_api import _ovewrite_named_param
except ImportError as e:
    _ovewrite_named_param = None


__all__ = [
    "VGG16",
    "VGG19",
    "VGG",
    "build_vgg",
]


cfgs: dict[str, list[Union[str, int]]] = {
    # 数字 = Conv2d 输出通道数, "M" = MaxPool2d(2,2) 空间尺寸减半
    # 以 3×224×224 输入为例，每经过一个 "M" 空间尺寸减半：224→112→56→28→14→7
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], # VGG11  (11层)
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], # VGG13  (13层)
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], # VGG16  (16层)
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], # VGG19  (19层)
}



def _make_layers(cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3×224×224 → 64×224×224 → Pool → 64×112×112
            # C1 : 112×112×64 
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/2

            # Block 2: 64×112×112 → 128×112×112 → Pool → 128×56×56
            # C2 : 56×56×128 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/4

            # Block 3: 128×56×56 → 256×56×56 → Pool → 256×28×28
            # C3 : 28×28×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/8

            # Block 4: 256×28×28 → 512×28×28 → Pool → 512×14×14
            # C4 : 14×14×512
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/16

            # Block 5: 512×14×14 → 512×14×14 → Pool → 512×7×7
            # C5 : 7×7×512
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/32
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # F6 nn.Linear(512, 4096) without self.avgpool
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),  # F7
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, num_classes) # F8
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (B, 512*7*7) = (B, 25088)
        x = self.classifier(x)
        return x



class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3×224×224 → 64×224×224 → Pool → 64×112×112
            # C1 : 112×112×64 
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/2

            # Block 2: 64×112×112 → 128×112×112 → Pool → 128×56×56
            # C2 : 56×56×128 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/4

            # Block 3: 128×56×56 → 256×56×56 → Pool → 256×28×28
            # C3 : 28×28×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/8

            # Block 4: 256×28×28 → 512×28×28 → Pool → 512×14×14
            # C4 : 14×14×512
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/16

            # Block 5: 512×14×14 → 512×14×14 → Pool → 512×7×7
            # C5 : 7×7×512
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/32
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # F6 nn.Linear(512, 4096) without self.avgpool
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),  # F7
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, num_classes) # F8
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (B, 512*7*7) = (B, 25088)
        x = self.classifier(x)
        return x



class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def _vgg(cfg: str, 
         batch_norm: bool=False, 
         weights: Optional[WeightsEnum]=None, 
         progress: bool=False, 
         **kwargs) -> VGG:
    if weights is not None and _ovewrite_named_param:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(_make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)
    return model



def build_vgg(arch:str ='vgg16', cfg:str ='D', 
              weights: Optional[WeightsEnum] = None, 
              progress: bool=True, **kwargs):
    return _vgg(cfg=cfg, batch_norm=False,  weights=weights, progress=progress, **kwargs)
 


if __name__ == "__main__":

    from torchvision.models.vgg import VGG16_Weights
    from torchinfo import summary

    model = build_vgg(weights= VGG16_Weights.DEFAULT)
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(4096, num_classes)   # 替换最后一层
    
    input_size = (1, 3, 224, 224)
    # dummy_data = torch.randn(input_size)
    summary(model, input_size=input_size)

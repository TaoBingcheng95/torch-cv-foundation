
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models import WeightsEnum, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

from .utils.pytorch_api import _ovewrite_named_param



__all__ = [
    "ResNet18",
    "ResNet50",
    "ResNet",
    "build_resnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, 
                     out_planes, 
                     kernel_size=1, 
                     stride=stride, 
                     bias=False)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



cfgs: dict[str, list[Union[str, int]]] = {
    "resnet18": [BasicBlock, [2, 2, 2, 2]],
    "resnet34": [BasicBlock, [3, 4, 6, 3]],
    "resnet50": [Bottleneck, [3, 4, 6, 3]], 
    "resnet101": [Bottleneck, [3, 4, 23, 3]], 
    "resnet152": [Bottleneck, [3, 8, 36, 3]]
}



class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



class ResNet18(nn.Module):
    """
    ResNet-18 硬编码实现（BasicBlock, [2, 2, 2, 2]）。

    结构展开便于初学者逐层阅读，与 ResNet(BasicBlock, [2,2,2,2]) 等价。
    输入: 3×224×224 → 输出: num_classes
    """
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        # Stem: 7×7 卷积 + BN + ReLU + 3×3 最大池化
        # 3×224×224 → 64×112×112 → Pool → 64×56×56
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1: 2×BasicBlock(64→64), 无降采样
        # 64×56×56 → 64×56×56
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64), # 恒等映射，无 downsample
            BasicBlock(64, 64)  # 恒等映射，无 downsample
        )

        # Layer 2: 2×BasicBlock(64→128, stride=2), 首个 block 降采样
        # 64×56×56 → 128×28×28
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2,
                       downsample=nn.Sequential(
                           conv1x1(64, 128, stride=2),
                           nn.BatchNorm2d(128))),
            BasicBlock(128, 128) # 恒等映射
        )

        # Layer 3: 2×BasicBlock(128→256, stride=2), 首个 block 降采样
        # 128×28×28 → 256×14×14
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2,
                       downsample=nn.Sequential(
                           conv1x1(128, 256, stride=2),
                           nn.BatchNorm2d(256))),
            BasicBlock(256, 256) # 恒等映射
        )

        # Layer 4: 2×BasicBlock(256→512, stride=2), 首个 block 降采样
        # 256×14×14 → 512×7×7
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2,
                       downsample=nn.Sequential(
                           conv1x1(256, 512, stride=2),
                           nn.BatchNorm2d(512))),
            BasicBlock(512, 512) # 恒等映射
        )

        # 分类头: 全局平均池化 + 全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.conv1(x)    # (B, 3, 224, 224) → (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, 112, 112) → (B, 64, 56, 56)

        # Layer 1: 64×56×56 → 64×56×56
        x = self.layer1(x)
        # Layer 2: 64×56×56 → 128×28×28
        x = self.layer2(x)
        # Layer 3: 128×28×28 → 256×14×14
        x = self.layer3(x)
        # Layer 4: 256×14×14 → 512×7×7
        x = self.layer4(x)
        # 分类头
        x = self.avgpool(x)          # (B, 512, 7, 7) → (B, 512, 1, 1)
        x = torch.flatten(x, 1)      # (B, 512)
        x = self.fc(x)               # (B, num_classes)
        return x



class ResNet50(nn.Module):
    """
    ResNet-50 硬编码实现（Bottleneck, [3, 4, 6, 3]）。

    结构展开便于初学者逐层阅读，与 ResNet(Bottleneck, [3,4,6,3]) 等价。
    注意：Bottleneck 的 expansion=4，输出通道数为 planes×4。
    输入: 3×224×224 → 输出: num_classes
    """
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        # Stem: 7×7 卷积 + BN + ReLU + 3×3 最大池化
        # 3×224×224 → 64×112×112 → Pool → 64×56×56
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1: 3×Bottleneck(64→64, expand=256), 首个 block 降采样通道
        # 64×56×56 → 256×56×56
        # self.layer1_block0 = Bottleneck(64, 64,
        #                                 downsample=nn.Sequential(
        #                                     conv1x1(64, 256),
        #                                     nn.BatchNorm2d(256)))
        # self.layer1_block1 = Bottleneck(256, 64)  # 恒等映射
        # self.layer1_block2 = Bottleneck(256, 64)  # 恒等映射
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 
                       downsample=nn.Sequential(
                           conv1x1(64, 256), 
                           nn.BatchNorm2d(256))),
            Bottleneck(256, 64), # 恒等映射
            Bottleneck(256, 64)  # 恒等映射
        )


        # Layer 2: 4×Bottleneck(256→128, expand=512, stride=2)
        # 256×56×56 → 512×28×28
        # self.layer2_block0 = Bottleneck(256, 128, stride=2,
        #                                 downsample=nn.Sequential(
        #                                     conv1x1(256, 512, stride=2),
        #                                     nn.BatchNorm2d(512)))
        # self.layer2_block1 = Bottleneck(512, 128)  # 恒等映射
        # self.layer2_block2 = Bottleneck(512, 128)  # 恒等映射
        # self.layer2_block3 = Bottleneck(512, 128)  # 恒等映射
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, stride=2, 
                    downsample=nn.Sequential(
                        conv1x1(256, 512, stride=2), 
                        nn.BatchNorm2d(512))),
            Bottleneck(512, 128),# 恒等映射
            Bottleneck(512, 128),# 恒等映射
            Bottleneck(512, 128)# 恒等映射
        )

        # Layer 3: 6×Bottleneck(512→256, expand=1024, stride=2)
        # 512×28×28 → 1024×14×14
        # self.layer3_block0 = Bottleneck(512, 256, stride=2,
        #                                 downsample=nn.Sequential(
        #                                     conv1x1(512, 1024, stride=2),
        #                                     nn.BatchNorm2d(1024)))
        # self.layer3_block1 = Bottleneck(1024, 256)  # 恒等映射
        # self.layer3_block2 = Bottleneck(1024, 256)  # 恒等映射
        # self.layer3_block3 = Bottleneck(1024, 256)  # 恒等映射
        # self.layer3_block4 = Bottleneck(1024, 256)  # 恒等映射
        # self.layer3_block5 = Bottleneck(1024, 256)  # 恒等映射
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, stride=2,
                       downsample=nn.Sequential(
                           conv1x1(512, 1024, stride=2), 
                           nn.BatchNorm2d(1024))),
            Bottleneck(1024, 256),# 恒等映射
            Bottleneck(1024, 256),# 恒等映射
            Bottleneck(1024, 256),# 恒等映射
            Bottleneck(1024, 256),# 恒等映射
            Bottleneck(1024, 256)
        )

        # Layer 4: 3×Bottleneck(1024→512, expand=2048, stride=2)
        # 1024×14×14 → 2048×7×7
        # self.layer4_block0 = Bottleneck(1024, 512, stride=2,
        #                                 downsample=nn.Sequential(
        #                                     conv1x1(1024, 2048, stride=2),
        #                                     nn.BatchNorm2d(2048)))
        # self.layer4_block1 = Bottleneck(2048, 512)  # 恒等映射
        # self.layer4_block2 = Bottleneck(2048, 512)  # 恒等映射
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=2,
                       downsample=nn.Sequential(
                           conv1x1(1024, 2048, stride=2), 
                           nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),# 恒等映射
            Bottleneck(2048, 512)# 恒等映射
        )

        # 分类头: 全局平均池化 + 全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.conv1(x)    # (B, 3, 224, 224) → (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, 112, 112) → (B, 64, 56, 56)

        # Layer 1: 64×56×56 → 256×56×56
        x = self.layer1(x)

        # Layer 2: 256×56×56 → 512×28×28
        x = self.layer2(x)

        # Layer 3: 512×28×28 → 1024×14×14
        x = self.layer3(x)

        # Layer 4: 1024×14×14 → 2048×7×7
        x = self.layer4(x)

        # 分类头
        x = self.avgpool(x)          # (B, 2048, 7, 7) → (B, 2048, 1, 1)
        x = torch.flatten(x, 1)      # (B, 2048)
        x = self.fc(x)               # (B, num_classes)
        return x



def _resnet(
    block: type[Union[BasicBlock, Bottleneck]],
    layers: list[int],
    weights: Optional[WeightsEnum]=None,
    progress: bool=False,
    **kwargs: Any,
    ) -> ResNet:
    if weights is not None and _ovewrite_named_param:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model



def build_resnet(arch:str ='resnet50', cfg:str ='resnet50', 
                 weights: Optional[WeightsEnum] = None, 
                 progress: bool=True, **kwargs):
    block, layers = cfgs.get(cfg, 'resnet50')
    return _resnet(block, layers, weights=weights, progress=progress, **kwargs)



def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)



def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)



def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """

    weights = ResNet101_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """

    weights = ResNet152_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)




if __name__ == "__main__":

    from torchinfo import summary

    model = ResNet18()
    input_size = (1, 3, 224, 224)
    # dummy_data = torch.randn(input_size)
    summary(model, input_size=input_size)
    
    

    # # ResNet18
    # m1 = ResNet18(num_classes=10)
    # m2 = resnet18(num_classes=10)
    # m1.load_state_dict(m2.state_dict())
    # m1.eval(); 
    # m2.eval()
    # with torch.no_grad():
    #     diff18 = (m1(x) - m2(x)).abs().max().item()
    # print(f"ResNet18 差异: {diff18:.2e}")  # 应为 0.00e+00

    # # ResNet50
    # m1 = ResNet50(num_classes=10)
    # m2 = resnet50(num_classes=10)
    # m1.load_state_dict(m2.state_dict())
    # m1.eval(); 
    # m2.eval()
    # with torch.no_grad():
    #     diff50 = (m1(x) - m2(x)).abs().max().item()
    # print(f"ResNet50 差异: {diff50:.2e}")  # 应为 0.00e+00

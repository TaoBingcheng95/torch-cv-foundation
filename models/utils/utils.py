# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common trainer utilities."""

import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import cast, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d, Module
# from torchvision.models._api import WeightsEnum


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    训练时以 drop_prob 概率随机丢弃整个残差分支，推理时恒等映射。

    与 timm.models.layers.DropPath / torchvision.ops.StochasticDepth 等价。
    使用 floor(rand + keep_prob) 代替 bernoulli，避免显式除法采样。

    Args:
        drop_prob: 丢弃概率 (0.0 = 不丢弃, 1.0 = 全部丢弃)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"



def extract_backbone(path: str) -> tuple[str, 'OrderedDict[str, Tensor]']:
    """
    Extracts a backbone from a lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)

    Returns:
        tuple containing model name and state dict

    Raises:
        ValueError: if 'model' or 'backbone' not in
            checkpoint['hyper_parameters']

    .. versionchanged:: 0.4
        Renamed from *extract_encoder* to *extract_backbone*
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if 'model' in checkpoint['hyper_parameters']:
        name = checkpoint['hyper_parameters']['model']
        state_dict = checkpoint['state_dict']
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if 'model.' in k})
        state_dict = OrderedDict(
            {k.replace('model.', ''): v for k, v in state_dict.items()}
        )
    elif 'backbone' in checkpoint['hyper_parameters']:
        name = checkpoint['hyper_parameters']['backbone']
        state_dict = checkpoint['state_dict']
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if 'model.backbone.model' in k}
        )
        state_dict = OrderedDict(
            {k.replace('model.backbone.model.', ''): v for k, v in state_dict.items()}
        )
    else:
        raise ValueError(
            'Unknown checkpoint task. Only backbone or model extraction is supported'
        )

    return name, state_dict


def _get_input_layer_name_and_module(model: Module) -> tuple[str, Module]:
    """
    Retrieve the input layer name and modules from a timm model.

    Args:
        model: timm model
    """
    keys = []
    module = None
    children = list(model.named_children())
    while children:
        name, module = children[0]
        keys.append(name)
        children = list(module.named_children())

    key = '.'.join(keys)
    return key, module


def load_state_dict(model: Module, state_dict: 'OrderedDict[str, Tensor]') -> tuple[list[str], list[str]]:
    """
    Load pretrained resnet weights to a model.

    Args:
        model: model to load the pretrained weights to
        state_dict: dict containing tensor parameters

    Returns:
        The missing and unexpected keys

    Warns:
        If input channels in model != pretrained model input channels
        If num output classes in model != pretrained model num classes
    """
    input_module_key, input_module = _get_input_layer_name_and_module(model)
    in_channels = input_module.in_channels
    expected_in_channels = state_dict[input_module_key + '.weight'].shape[1]

    output_module_key, output_module = list(model.named_children())[-1]
    if isinstance(output_module, nn.Identity):
        num_classes = model.num_features
    else:
        num_classes = output_module.out_features
    expected_num_classes = None
    if output_module_key + '.weight' in state_dict:
        expected_num_classes = state_dict[output_module_key + '.weight'].shape[0]

    if in_channels != expected_in_channels:
        warnings.warn(
            f'input channels {in_channels} != input channels in pretrained'
            f' model {expected_in_channels}. Overriding with new input channels'
        )
        del state_dict[input_module_key + '.weight']

    if expected_num_classes and num_classes != expected_num_classes:
        warnings.warn(
            f'num classes {num_classes} != num classes in pretrained model'
            f' {expected_num_classes}. Overriding with new num classes'
        )
        del (
            state_dict[output_module_key + '.weight'],
            state_dict[output_module_key + '.bias'],
        )

    missing_keys: list[str]
    unexpected_keys: list[str]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


def reinit_initial_conv_layer(
        layer: Conv2d,
        new_in_channels: int,
        keep_rgb_weights: bool,
        new_stride: int | tuple[int, int] | None = None,
        new_padding: str | int | tuple[int, int] | None = None,) -> Conv2d:
    """
    Clones a Conv-2d layer while optionally retaining some of the original weights.

    When replacing the first convolutional layer in a model with one that operates over
    different number of input channels, we sometimes want to keep a subset of the kernel
    weights the same (e.g. the RGB weights of an ImageNet pretrained model). This is a
    convenience function that performs that function.

    Args:
        layer: the Conv-2d layer to initialize
        new_in_channels: the new number of input channels
        keep_rgb_weights: flag indicating whether to re-initialize the first 3 channels
        new_stride: optionally, overwrites the ``layer``'s stride with this value
        new_padding: optionally, overwrites the ``layers``'s padding with this value

    Returns:
        a Conv-2d layer with new kernel weights
    """
    use_bias = layer.bias is not None
    w_old = None
    b_old = None
    if keep_rgb_weights:
        w_old = layer.weight.data[:, :3, :, :].clone()
        if use_bias:
            b_old = cast(Tensor, layer.bias).data.clone()

    updated_stride = layer.stride if new_stride is None else new_stride
    updated_padding = layer.padding if new_padding is None else new_padding

    new_layer = Conv2d(
        new_in_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,  # type: ignore[arg-type]
        stride=updated_stride,  # type: ignore[arg-type]
        padding=updated_padding,  # type: ignore[arg-type]
        dilation=layer.dilation,  # type: ignore[arg-type]
        groups=layer.groups,
        bias=use_bias,
        padding_mode=layer.padding_mode,
    )
    nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')

    if keep_rgb_weights:
        new_layer.weight.data[:, :3, :, :] = w_old
        if use_bias:
            cast(Tensor, new_layer.bias).data = b_old

    return new_layer


def modify_resnet(model,in_channels=3):
    # import torchvision.models as models
    # backbone = model.resnet18()
    backbone = model.resnet18()
    original_conv1 = backbone.conv1
    backbone.conv1 = nn.Conv2d(in_channels, original_conv1.out_channels, 
                                kernel_size=original_conv1.kernel_size,
                                stride=original_conv1.stride, 
                                padding=original_conv1.padding, 
                                bias=original_conv1.bias is not None)
    # model.resnet18() = backbone
    return backbone

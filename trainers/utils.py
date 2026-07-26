"""
Common trainer utilities
"""

import os
import warnings
from collections import OrderedDict
from typing import cast, Any, Dict, List, Optional, Tuple
import json
# import logging
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d, Module
import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler



@dataclass
class TrainConfig:
    max_epochs: int = 30
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    milestones: tuple = (20, 25)
    gamma: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 5
    eval_interval: int = 5   # 每 N 个 epoch 评估一次；默认 5，30 epoch 共 6 次评估
    log_interval: int = 50  # 每 N 个 batch 打印一次；默认 50，减少高频 IO
    work_dir: str = './checkpoints/tracknetv2'
    monitor_metric: str = 'PCK@0.10'  # 监控指标：越大越好
    monitor_mode: str = 'max'
    gpus: int = 1
    ddp_port: int = 29500
    ddp_backend: str = 'nccl'



class History:
    def __init__(self, log_path, max_memory_records=5000):
        self.log_path = Path(log_path)
        self.records = []
        self._max_memory_records = max_memory_records # 防止内存溢出
        self._file_handle = None

    def _get_file_handle(self):
        if self._file_handle is None or self._file_handle.closed:
            # 确保目录存在
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_path, 'a', encoding='utf-8')
        return self._file_handle

    def append(self, record):
        # 写入磁盘并 flush (保证崩溃时数据不丢)
        fh = self._get_file_handle()
        fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        fh.flush()
        
        # 内存管理：只保留最近的 N 条记录，用于实时绘图或监控
        # 如果需要全部历史，可以通过 load() 从磁盘重新读取
        self.records.append(record)
        if len(self.records) > self._max_memory_records:
            # 移除最旧的记录 (FIFO)
            self.records.pop(0)

    def close(self):
        """显式关闭文件句柄，比依赖 __del__ 更安全"""
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        # 作为最后一道防线
        self.close()

    def get(self, key, phase='train'):
        # 注意：如果训练步数极多且超出了 max_memory_records，
        # 这里只能获取到最近的记录。对于绘图（通常按 epoch 聚合）来说完全足够。
        return [r[key] for r in self.records if r.get('phase') == phase and key in r]

    def load(self):
        """从磁盘完整加载所有历史记录（用于断点续训后的绘图）"""
        if not self.log_path.exists():
            return
        
        # 修复：统一使用 utf-8 编码
        with open(self.log_path, 'r', encoding='utf-8') as f:
            self.records = [json.loads(line) for line in f if line.strip()]




class EarlyStopping:
    """
    早停机制：当验证损失连续 patience 个 epoch 不再改善时提前停止训练。

    职责单一：只负责判断训练是否继续，不涉及模型保存
    （检查点保存由 Trainer 在 fit 中统一管理）。

    Attributes:
        patience: 容忍的 epoch 数，超过后停止训练
        delta: 损失改善的最小阈值
        verbose: 是否打印详细信息
    """
    def __init__(self,
                 patience: int = 10,
                 delta: float = 0.0,
                 verbose: bool = False,
                 ):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, epoch: int) -> None:
        """
        检查是否需要早停
        
        Args:
            val_loss: 验证集损失
            epoch: 当前 epoch
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⚠️ EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'🛑 Early stopping triggered at epoch {epoch}')
        else:
            if self.verbose:
                print(f'✨ Validation loss improved: {self.val_loss_min:.6f} → {val_loss:.6f}')
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0





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
    children = list(model.named_children())
    while children:
        name, module = children[0]
        keys.append(name)
        children = list(module.named_children())

    key = '.'.join(keys)
    return key, module


def load_state_dict(
    model: Module, state_dict: 'OrderedDict[str, Tensor]') -> tuple[list[str], list[str]]:
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
    Clones a Conv2d layer while optionally retaining some of the original weights.

    When replacing the first convolutional layer in a model with one that operates over
    different number of input channels, we sometimes want to keep a subset of the kernel
    weights the same (e.g. the RGB weights of an ImageNet pretrained model). This is a
    convenience function that performs that function.

    Args:
        layer: the Conv2d layer to initialize
        new_in_channels: the new number of input channels
        keep_rgb_weights: flag indicating whether to re-initialize the first 3 channels
        new_stride: optionally, overwrites the ``layer``'s stride with this value
        new_padding: optionally, overwrites the ``layers``'s padding with this value

    Returns:
        a Conv2d layer with new kernel weights
    """
    use_bias = layer.bias is not None
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



def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size



def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


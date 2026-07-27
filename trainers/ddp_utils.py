"""
DDP 工具集：进程组初始化/销毁、rank 查询、跨进程聚合与广播
"""

import os
from datetime import timedelta
from typing import Any, Dict, Union

import torch
from torch import distributed as dist



__all__ = [
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'get_world_size',
    'get_rank',
    'barrier',
    'reduce_value',
    'reduce_dict',
    'broadcast_flag',
    '_NoopHistory', 
    '_NoopVisualizer', 
    '_DistributedMetrics'
]



class _NoopHistory:
    """非主进程的历史记录器替身：不落盘，接口与 History 对齐"""

    def append(self, record: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class _NoopVisualizer:
    """非主进程的可视化器替身：所有绘图/报告调用静默吾临，避免多进程重复写文件"""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _DistributedMetrics:
    """
    分布式指标包装器：compute() 前先 all_reduce 混淆矩阵，
    使所有 rank 得到完全一致的全局指标（集合通信保证各 rank 结果相同，
    因此 best.pt 保存/早停决策在各 rank 间天然同步，无需额外广播）。

    注意：compute() 包含集合通信，所有 rank 必须同步调用相同次数。
    """

    def __init__(self, metrics):
        self.metrics = metrics
        self._synced = False  # 防止同一轮重复 compute 时混淆矩阵被叠加多次

    def reset(self) -> None:
        self._synced = False
        self.metrics.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._synced = False
        self.metrics.update(preds, targets)

    @property
    def confusion_matrix(self) -> torch.Tensor:
        return self.metrics.confusion_matrix

    def compute(self) -> Dict[str, float]:
        if not self._synced and get_world_size() > 1:
            global_cm = reduce_value(self.metrics.confusion_matrix, average=False)
            self.metrics.confusion_matrix.copy_(global_cm)
            self._synced = True
        return self.metrics.compute()





def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier() -> None:
    """同步点安全包装：单进程/未初始化时为 no-op，可无条件调用"""
    if get_world_size() > 1:
        dist.barrier()


def _collective_device() -> torch.device:
    """集合通信用的设备：NCCL 要求 CUDA tensor，gloo 用 CPU tensor"""
    if dist.get_backend() == 'nccl':
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def reduce_value(value: Union[torch.Tensor, float, int],
                 average: bool = True) -> Union[torch.Tensor, float]:
    """
    跨进程 all_reduce（求和或均值）

    - 兼容 python 数值和 tensor：数值入 → float 出；tensor 入 → tensor 出
    - 不原地修改调用方的 tensor（内部 clone）
    - 自动搬到通信后端要求的设备（NCCL 需 CUDA tensor）
    """
    world_size = get_world_size()
    if world_size < 2:  # 单进程：原样返回
        return value

    is_number = not torch.is_tensor(value)
    device = _collective_device()
    with torch.no_grad():
        if is_number:
            t = torch.tensor(float(value), dtype=torch.float64, device=device)
        else:
            t = value.detach().clone().to(device)
        dist.all_reduce(t)   # 默认 SUM
        if average:
            t /= world_size
    return t.item() if is_number else t


def reduce_dict(input_dict: Dict[str, Any], average: bool = True) -> Dict[str, float]:
    """
    对指标 dict 做一次性 all_reduce（所有值拼成单个 tensor，只通信一次）

    要求所有进程以相同的键顺序调用；值可为 python 数值或 0-dim tensor。
    返回全 float 的新 dict，不修改入参。
    """
    world_size = get_world_size()
    if world_size < 2:
        return {k: float(v) for k, v in input_dict.items()}

    keys = list(input_dict.keys())
    values = torch.tensor([float(input_dict[k]) for k in keys],
                          dtype=torch.float64, device=_collective_device())
    with torch.no_grad():
        dist.all_reduce(values)
        if average:
            values /= world_size
    return {k: v.item() for k, v in zip(keys, values)}


def broadcast_flag(flag: bool, src: int = 0) -> bool:
    """
    从 src 进程广播布尔决策（如早停/is_best），保证所有 rank 行为一致，
    避免各 rank 浮点误差导致退出不同步而死锁
    """
    if get_world_size() < 2:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int64,
                     device=_collective_device())
    dist.broadcast(t, src=src)
    return bool(t.item())


def setup_distributed(timeout_minutes: int = 30) -> tuple[bool, int, int, int]:
    """
    初始化进程组（仅支持 torchrun 启动，env:// 方式）

    Returns:
        (distributed, rank, local_rank, world_size)；未经 torchrun 启动时
        返回 (False, 0, 0, 1)，调用方按单进程路径运行
    """
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        # 必须在 init_process_group 之前绑定本进程 GPU，
        # 否则 NCCL 可能让所有进程先落到 GPU0 上
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        timeout=timedelta(minutes=timeout_minutes),  # 显式超时，方便排查 rank 挂死
    )
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # 确保所有 rank 完成后再销毁，避免快进程提前退出
        dist.destroy_process_group()

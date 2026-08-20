"""
DDP 工具集：进程组初始化/销毁、rank 查询、跨进程聚合与广播
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

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
    """非主进程的历史记录器替身：不落盘，接口与 History 对齐

    基类 Trainer 的断点续训流程会调用 load()/records/get()，
    故这些接口也要给出安全的空实现，而不只是 append/close
    """

    records: List[Dict[str, Any]] = []
    log_path: Optional[Any] = None

    def append(self, record: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass

    def load(self) -> None:
        pass

    def get(self, key: str, phase: Optional[str] = None) -> List[Any]:
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __len__(self) -> int:
        return 0


class _NoopVisualizer:
    """非主进程的可视化器替身：所有绘图/报告调用静默跳过，避免多进程重复写文件"""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _DistributedMetrics:
    """
    分布式指标透传包装器。

    包装原生 PyTorch 指标类（ClassificationMetric / SegmentationMetric），
    在 compute() / per_class_metrics() / confusion_matrix 前自动调用
    all_reduce() 汇总各 rank 的混淆矩阵，确保指标为全局精确值。

    前提约束（由调用方保证）：
        - 指标类实现 all_reduce() 方法（内部调用 dist.all_reduce SUM）
        - 指标实例已 .to(通信设备)（NCCL 后端要求 state 在 CUDA）
        - 所有 rank 以相同顺序调用 update/compute（集合通信同步性要求）

    注意：compute() 含集合通信，所有 rank 必须同步调用相同次数。
    """

    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self) -> None:
        self.metrics.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metrics.update(preds, targets)

    def all_reduce(self) -> None:
        """显式触发跨 rank 混淆矩阵汇总（幂等：非 DDP 或已汇总时为 no-op）"""
        if hasattr(self.metrics, 'all_reduce'):
            self.metrics.all_reduce()

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """返回 all_reduce 后的全局混淆矩阵"""
        self.all_reduce()
        return self.metrics.confusion_matrix

    def compute(self) -> Dict[str, float]:
        """计算指标前先 all_reduce，确保结果为全局值"""
        self.all_reduce()
        return self.metrics.compute()

    def per_class_metrics(self) -> Dict[str, torch.Tensor]:
        """逐类指标（内部先 all_reduce）"""
        self.all_reduce()
        return self.metrics.per_class_metrics()





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
        dist.barrier(device_ids=_barrier_device_ids())


def _barrier_device_ids() -> Optional[List[int]]:
    """NCCL 下需显式告知 barrier 用哪张卡，否则 torch 会猜设备并告警"""
    if dist.get_backend() == 'nccl':
        return [torch.cuda.current_device()]
    return None


def _collective_device() -> torch.device:
    """集合通信用的设备：NCCL 要求 CUDA tensor，gloo 用 CPU tensor"""
    if not (dist.is_available() and dist.is_initialized()):
        return torch.device('cpu')
    if dist.get_backend() == 'nccl':
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def reduce_value(value: Union[torch.Tensor, float, int],
                 average: bool = True) -> Union[torch.Tensor, float]:
    """
    跨进程 all_reduce（求和或均值）

    - 兼容 python 数值和 tensor：数值入 → float 出；tensor 入 → tensor 出
    - 不原地修改调用方的 tensor（内部 clone）
    - 自动移到通信后端要求的设备（NCCL 需 CUDA tensor）
    - 单进程与多进程走相同的类型规范，调用方无需区分单卡/多卡

    Note:
        average=True 会把整型 tensor 提升为 float64（整型无法做真除法）。
        计数类状态应使用 average=False 以保留整型 dtype。
    """
    world_size = get_world_size()
    is_number = not torch.is_tensor(value)
    if world_size < 2:
        # 单进程也做同样的类型规范，避免调用方在单卡/多卡下拿到不同类型
        return float(value) if is_number else value.detach().clone()

    device = _collective_device()
    with torch.no_grad():
        if is_number:
            t = torch.tensor(float(value), dtype=torch.float64, device=device)
        else:
            t = value.detach().clone().to(device)
            if average and not t.is_floating_point():
                # 整型 tensor 无法就地真除（RuntimeError），先提升再求均值；
                # average 在各 rank 一致，因此通信 dtype 仍然各 rank 相同
                t = t.to(torch.float64)
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

    幂等：进程组已存在时直接复用（重复调用 / 测试 / notebook 场景），
    而不是让 init_process_group 报错。

    Returns:
        (distributed, rank, local_rank, world_size)；未经 torchrun 启动时
        返回 (False, 0, 0, 1)，调用方按单进程路径运行
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        return True, rank, int(os.environ.get("LOCAL_RANK", rank)), dist.get_world_size()

    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torchrun 一定会设 WORLD_SIZE；容错取值避免在非标准启动器下 KeyError
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} 超出可见 GPU 数量 {device_count}，"
                f"请检查 --nproc_per_node 与 CUDA_VISIBLE_DEVICES 是否匹配"
            )
        # 必须在 init_process_group 之前绑定本进程 GPU，
        # 否则 NCCL 可能让所有进程先落到 GPU0 上
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        timeout=timedelta(minutes=timeout_minutes),  # 显式超时，方便排查 rank 挂死
    )
    return True, rank, local_rank, world_size


def _bounded_barrier(timeout_seconds: float) -> bool:
    """
    有界同步：返回所有 rank 是否都到达（不抛异常，避免清理路径掩盖主流程异常）

    gloo 优先用 monitored_barrier：它是同步实现，超时即报错并指出缺席的 rank，
    且不会在进程组里留下未完成的 work（NCCL 不支持该 API，因此回退到
    异步 barrier + 有界轮询）。
    """
    try:
        if dist.get_backend() == 'gloo':
            dist.monitored_barrier(timeout=timedelta(seconds=timeout_seconds))
            return True
        work = dist.barrier(device_ids=_barrier_device_ids(), async_op=True)
        deadline = time.monotonic() + timeout_seconds
        while not work.is_completed():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True
    except Exception as exc:
        warnings.warn(f"cleanup_distributed: 退出前同步失败（{exc}）", RuntimeWarning)
        return False


def cleanup_distributed(timeout_seconds: float = 60.0) -> None:
    """
    销毁进程组；退出前的同步等待有界，不会因某个 rank 已崩溃而长时间挂住

    正常路径下等所有 rank 到达后再销毁，避免快进程提前退出。异常路径下往往只有
    部分 rank 走到清理，此时阻塞式 barrier 会一直挂到进程组超时（默认 30 分钟），
    反而掩盖真正的异常栈，因此这里改用有界同步。

    Args:
        timeout_seconds: 退出前等待其余 rank 的上限；<=0 表示不等待直接销毁

    Note:
        同步超时时会告警并**跳过** destroy_process_group：此时进程组内可能还挂着
        未完成的集合通信，destroy 会一直等其完成（实测会阻塞到进程组超时），
        而进程组资源本身会随进程退出由系统回收。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if get_world_size() > 1 and timeout_seconds > 0 and not _bounded_barrier(timeout_seconds):
        warnings.warn(
            "cleanup_distributed: 部分 rank 未到达同步点（通常意味着某个 rank 已异常退出），"
            "跳过 destroy_process_group 以免阻塞退出；进程组随进程结束由系统回收",
            RuntimeWarning,
        )
        return
    dist.destroy_process_group()

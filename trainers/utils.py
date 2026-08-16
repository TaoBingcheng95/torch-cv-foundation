"""
Common trainer utilities
"""

import csv
import logging
import os
import warnings
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator


@dataclass
class TrainConfig:
    """
    训练配置数据类，集中管理训练流程的全部可调参数。

    设计原则：
      - 优化器/调度器参数分别聚合为 optim_cfg / sched_cfg 字典，
        与 build_optimizer / build_scheduler 的 cfg 入参同构，可直接传递
      - 训练控制参数（epochs/eval/monitor/早停等）独立于组件配置
      - DDP/硬件参数供分布式训练使用

    使用示例::

        cfg = TrainConfig(max_epochs=30)
        optimizer = build_optimizer(model, cfg.optim_cfg)
        scheduler = build_scheduler(optimizer, cfg.sched_cfg, total_epochs=cfg.max_epochs)
    """
    # ── 训练控制 ────────────────────────────────────────────────────────────
    max_epochs: int = 30
    batch_size: int = 16
    num_workers: int = 4
    eval_interval: int = 1        # 每 N 个 epoch 验证一次（1 = 每轮都验证）
    log_interval: int = 5         # 每 N 个 batch 打印一次日志
    grad_clip_norm: Optional[float] = 1.0  # 梯度裁剪（None 表示禁用）
    work_dir: str = './checkpoints'

    # ── 优化器配置（与 build_optimizer cfg 同构）──────────────────────────
    optim_cfg: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'adamw',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
    })

    # ── 调度器配置（与 build_scheduler cfg 同构）──────────────────────────
    # 注：total_epochs 在运行时由 max_epochs 填充，此处不预设
    sched_cfg: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'warmup_cosine',
        'warmup_epochs': 5,
    })

    # ── 监控与早停 ──────────────────────────────────────────────────────────
    monitor: str = 'val/acc'      # 统一监控指标（slash 前缀风格）
    monitor_mode: str = 'max'     # 'auto' | 'min' | 'max'
    early_stop_patience: Optional[int] = 5   # None 表示禁用早停
    early_stop_delta: float = 0.01           # 早停最小改善阈值
    min_epochs: int = 0                      # 最小训练轮数（早停保护期内不触发）

    # ── DDP / 硬件 ──────────────────────────────────────────────────────────
    gpus: int = 1
    ddp_port: int = 29500
    ddp_backend: str = 'nccl'



class History:
    """
    训练历史的 CSV 持久化器（Lightning 风格）。

    - 表头在首次写入时一次性写入（utf-8，Excel 友好）
    - 每行用 csv.DictWriter 追加 + flush，崩溃安全
    - 缺失字段填空字符串；未知字段忽略并打 warning（避免静默丢数据）
    - 内存 deque self.records 保留最近 dict 记录，供实时绘图查询
    - 支持上下文管理器（with 语法），自动关闭文件句柄

    Args:
        log_path: CSV 文件路径
        fieldnames: CSV 表头列名列表（必须预先确定，不演变）
        max_memory_records: 内存中保留的最近记录数上限，超出后 FIFO

    Example::

        with History('metrics.csv', fieldnames=['epoch', 'phase', 'train/loss']) as h:
            h.append({'epoch': 1, 'phase': 'train', 'train/loss': 0.5})
            print(len(h))  # 1
    """
    def __init__(self, log_path, fieldnames: List[str], max_memory_records: int = 5000):
        self.log_path = Path(log_path)
        self.fieldnames = list(fieldnames)
        self._fieldset = set(self.fieldnames)  # 快速查重
        self._max_memory_records = max_memory_records
        self.records: deque = deque(maxlen=max_memory_records)
        self._file_handle = None
        self._writer = None
        self._header_written = False
        self._warned_unknown = False

    def _get_file_handle(self):
        if self._file_handle is None or self._file_handle.closed:
            # 确保目录存在
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # utf-8-sig：写表头时带 BOM，Excel 直接双击不乱码；
            # 后续追加用普通 utf-8 也无妨，但保持同一句柄即可
            need_header = (not self.log_path.exists()) or self.log_path.stat().st_size == 0
            self._file_handle = open(self.log_path, 'a', encoding='utf-8', newline='')
            self._writer = csv.DictWriter(
                self._file_handle,
                fieldnames=self.fieldnames,
                restval='',                 # 缺失字段填空
                extrasaction='ignore',      # 未知字段忽略（已在 append 中打 warning）
            )
            if need_header:
                self._writer.writeheader()
                self._file_handle.flush()
            self._header_written = True
        return self._writer

    def append(self, record: Dict[str, Any]):
        # 写入前检查是否有未声明的字段（fail-fast 的弱化版：warn 不丢）
        unknown = set(record.keys()) - self._fieldset
        if unknown and not self._warned_unknown:
            warnings.warn(
                f"History 收到未在 fieldnames 中的字段 {sorted(unknown)}，"
                f"已忽略（后续相同情况不再提示）"
            )
            self._warned_unknown = True

        writer = self._get_file_handle()
        writer.writerow(record)
        self._file_handle.flush()

        # 内存管理：deque(maxlen=N) 自动 FIFO 淘汰旧记录，O(1)
        # 如果需要全部历史，可以通过 load() 从磁盘重新读取
        self.records.append(dict(record))

    def close(self):
        """显式关闭文件句柄，比依赖 __del__ 更安全"""
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # 不吞异常

    def __del__(self):
        # 作为最后一道防线
        self.close()

    def __len__(self) -> int:
        """返回内存中当前保留的记录数（≤ max_memory_records）"""
        return len(self.records)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """按时间顺序迭代内存中的记录"""
        return iter(self.records)

    def get(self, key: str, phase: Optional[str] = None) -> List[Any]:
        """
        从内存记录中提取指定列的值列表。

        Args:
            key: 列名（如 'train/loss'、'val/acc'、'epoch'）
            phase: 可选过滤条件（'train' / 'val' / None）。
                   None 时返回所有记录中该 key 的值（不过滤 phase）。

        Returns:
            符合条件的值列表（按时间顺序）

        Note:
            如果训练步数极多且超出了 max_memory_records，
            这里只能获取到最近的记录。对于绘图（通常按 epoch 聚合）来说完全足够。
        """
        if phase is None:
            return [r[key] for r in self.records if key in r]
        return [r[key] for r in self.records
                if r.get('phase') == phase and key in r]

    def load(self):
        """
        从磁盘完整加载所有历史记录（用于断点续训后的绘图）。

        CSV 读取后需做类型恢复：数字列转回 float/int，其余保持字符串。
        epoch 列强制 int，phase 强制 str，其余列尝试 float，失败则保留原值。

        加载后同步更新内部状态：
        - records 替换为磁盘全量数据（截断到 max_memory_records）
        - _header_written 置 True（文件已存在且有表头，后续 append 不会重写）
        """
        if not self.log_path.exists():
            return

        records: List[Dict[str, Any]] = []
        with open(self.log_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            int_keys = {'epoch'}
            for row in reader:
                typed: Dict[str, Any] = {}
                for k, v in row.items():
                    if v is None or v == '':
                        continue  # 空单元格不入 dict，与 append 时缺失字段语义一致
                    if k in int_keys:
                        try:
                            typed[k] = int(float(v))
                        except (ValueError, TypeError):
                            typed[k] = v
                    else:
                        try:
                            typed[k] = float(v)
                        except (ValueError, TypeError):
                            typed[k] = v
                records.append(typed)

        # 替换内存记录：deque 赋值后自动截断到 maxlen
        self.records = deque(records, maxlen=self._max_memory_records)
        # 文件已存在且有内容，后续 append 不应重写表头
        self._header_written = True



class EarlyStopping:
    """
    早停机制：监控指标连续 patience 次验证不再改善时提前停止训练。

    Attributes:
        patience: 容忍的验证次数，超过后停止训练
        delta: 指标改善的最小阈值
        verbose: 是否打印详细信息
        mode: 'min' 指标越小越好（如 val/loss），'max' 越大越好（如 val/acc）
        monitor_name: 监控指标名称（仅用于 verbose 打印，如 'val/loss'）

    Example::

        early_stop = EarlyStopping(patience=5, mode='max', monitor_name='val/acc')
        for epoch in range(max_epochs):
            ...
            early_stop(val_acc, epoch=epoch)
            if early_stop.early_stop:
                break
    """

    def __init__(self,
                 patience: int = 10,
                 delta: float = 0.0,
                 verbose: bool = False,
                 mode: str = 'min',
                 monitor_name: str = '',
                 ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
        self.monitor_name = monitor_name
        self._logger = logging.getLogger(__name__)
        self.reset()

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置所有运行时状态，保留配置参数（patience / delta / mode 等）。"""
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_value: Optional[float] = None  # 历史最佳的原始指标值
        self.early_stop = False

    def state_dict(self) -> Dict[str, Any]:
        """
        导出运行时状态，用于写入 checkpoint。

        Returns:
            包含 counter / best_score / best_value / early_stop 的字典
        """
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_value': self.best_value,
            'early_stop': self.early_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        从 checkpoint 恢复运行时状态（断点续训时调用）。

        Args:
            state: 由 state_dict() 导出的字典
        """
        self.counter = state.get('counter', 0)
        self.best_score = state.get('best_score')
        self.best_value = state.get('best_value')
        self.early_stop = state.get('early_stop', False)
        if self.verbose and self.best_value is not None:
            self._logger.info(
                f"EarlyStopping restored | best {self.monitor_name or 'metric'}: "
                f"{self.best_value:.6f} (counter={self.counter})"
            )

    # ------------------------------------------------------------------
    # 核心判定
    # ------------------------------------------------------------------

    def __call__(self, value: float, epoch: int) -> None:
        """
        检查是否需要早停

        Args:
            value: 当前监控指标值（按 mode 判断改善方向）
            epoch: 当前 epoch
        """
        # 统一转为"越大越好"的分数比较
        score = -value if self.mode == 'min' else value
        name = self.monitor_name or 'metric'

        if self.best_score is None:
            self.best_score = score
            self.best_value = value
        elif score <= self.best_score + self.delta:
            # 未改善（含持平：指标相等不算改善，否则指标停滞时永远不会触发早停）
            self.counter += 1
            if self.verbose:
                self._logger.info(
                    f'EarlyStopping [{name}] counter: {self.counter} / {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self._logger.info(
                        f'Early stopping [{name}] triggered at epoch {epoch}'
                    )
        else:
            if self.verbose:
                self._logger.info(
                    f'{name} improved: {self.best_value:.6f} → {value:.6f}'
                )
            self.best_score = score
            self.best_value = value
            self.counter = 0

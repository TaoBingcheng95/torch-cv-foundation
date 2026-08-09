"""
Common trainer utilities
"""

import csv
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    """
    训练历史的 CSV 持久化器（Lightning 风格）。

    - 表头在首次写入时一次性写入（utf-8-sig，Excel 友好）
    - 每行用 csv.DictWriter 追加 + flush，崩溃安全
    - 缺失字段填空字符串；未知字段忽略并打 warning（避免静默丢数据）
    - 内存列表 self.records 仍保留 dict 形式，供实时绘图查询

    Args:
        log_path: CSV 文件路径
        fieldnames: CSV 表头列名列表（必须预先确定，不演变）
        max_memory_records: 内存中保留的最近记录数上限，超出后 FIFO
    """
    def __init__(self, log_path, fieldnames: List[str], max_memory_records: int = 5000):
        self.log_path = Path(log_path)
        self.fieldnames = list(fieldnames)
        self._fieldset = set(self.fieldnames)  # 快速查重
        self._max_memory_records = max_memory_records
        self.records: List[Dict[str, Any]] = []
        self._file_handle = None
        self._writer = None
        self._header_written = False

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
        if unknown:
            # 保留首条 warning，避免相同字段反复刷屏
            if not getattr(self, '_warned_unknown', False):
                import warnings
                warnings.warn(
                    f"History 收到未在 fieldnames 中的字段 {sorted(unknown)}，"
                    f"已忽略（后续相同情况不再提示）"
                )
                self._warned_unknown = True

        writer = self._get_file_handle()
        writer.writerow(record)
        self._file_handle.flush()

        # 内存管理：只保留最近的 N 条记录，用于实时绘图或监控
        # 如果需要全部历史，可以通过 load() 从磁盘重新读取
        self.records.append(dict(record))
        if len(self.records) > self._max_memory_records:
            self.records.pop(0)

    def close(self):
        """显式关闭文件句柄，比依赖 __del__ 更安全"""
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None
            self._writer = None

    def __del__(self):
        # 作为最后一道防线
        self.close()

    def get(self, key: str, phase: str = 'train') -> List[Any]:
        # 注意：如果训练步数极多且超出了 max_memory_records，
        # 这里只能获取到最近的记录。对于绘图（通常按 epoch 聚合）来说完全足够。
        return [r[key] for r in self.records
                if r.get('phase') == phase and key in r]

    def load(self):
        """
        从磁盘完整加载所有历史记录（用于断点续训后的绘图）。

        CSV 读取后需做类型恢复：数字列转回 float/int，其余保持字符串。
        epoch 列强制 int，phase 强制 str，其余列尝试 float，失败则保留原值。
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
        self.records = records



class EarlyStopping:
    """
    早停机制：监控指标连续 patience 次验证不再改善时提前停止训练。
    Attributes:
        patience: 容忍的验证次数，超过后停止训练
        delta: 指标改善的最小阈值
        verbose: 是否打印详细信息
        mode: 'min' 指标越小越好（如 val/loss），'max' 越大越好（如 val/acc）
        monitor_name: 监控指标名称（仅用于 verbose 打印，如 'val/loss'）
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
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
        self.monitor_name = monitor_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = None  # 历史最佳的原始指标值

    def __call__(self, value: float, epoch: int) -> None:
        """
        检查是否需要早停

        Args:
            value: 当前监控指标值（按 mode 判断改善方向）
            epoch: 当前 epoch
        """
        # 统一转为“越大越好”的分数比较
        score = -value if self.mode == 'min' else value
        name = self.monitor_name or 'metric'

        if self.best_score is None:
            self.best_score = score
            self.best_value = value
        elif score <= self.best_score + self.delta:
            # 未改善（含持平：指标相等不算改善，否则指标停滞时永远不会触发早停）
            self.counter += 1
            if self.verbose:
                print(f'⚠️ EarlyStopping [{name}] counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'🛑 Early stopping [{name}] triggered at epoch {epoch}')
        else:
            if self.verbose:
                print(f'✨ {name} improved: {self.best_value:.6f} → {value:.6f}')
            self.best_score = score
            self.best_value = value
            self.counter = 0

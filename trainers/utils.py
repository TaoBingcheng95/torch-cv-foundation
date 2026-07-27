"""
Common trainer utilities
"""

import os
import json
# import logging
from pathlib import Path
from dataclasses import dataclass, field


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
    早停机制：监控指标连续 patience 次验证不再改善时提前停止训练。
    Attributes:
        patience: 容忍的验证次数，超过后停止训练
        delta: 指标改善的最小阈值
        verbose: 是否打印详细信息
        mode: 'min' 指标越小越好（如 loss），'max' 越大越好（如 acc）
    """
    def __init__(self,
                 patience: int = 10,
                 delta: float = 0.0,
                 verbose: bool = False,
                 mode: str = 'min',
                 ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
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
        
        if self.best_score is None:
            self.best_score = score
            self.best_value = value
        elif score <= self.best_score + self.delta:
            # 未改善（含持平：指标相等不算改善，否则指标停滞时永远不会触发早停）
            self.counter += 1
            if self.verbose:
                print(f'⚠️ EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'🛑 Early stopping triggered at epoch {epoch}')
        else:
            if self.verbose:
                print(f'✨ Monitored metric improved: {self.best_value:.6f} → {value:.6f}')
            self.best_score = score
            self.best_value = value
            self.counter = 0

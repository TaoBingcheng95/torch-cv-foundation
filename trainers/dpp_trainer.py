# ============================================================
# DDP 支持：DDPTrainer
# ============================================================

import os
from pathlib import Path
import logging
from typing import Dict, Optional, Any

import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .logger import get_logger
from .ddp_utils import (
    setup_distributed, cleanup_distributed, barrier,
    reduce_value, _NoopHistory, _NoopVisualizer, _DistributedMetrics
)
from metrics import Metrics
from .basetrainer import BaseTrainer

# 日志配置
# import logging
# logging.basicConfig(level=logging.INFO, 
#                     format="%(asctime)s - %(levelname)s - %(message)s",
#                     datefmt="%Y-%m-%d %H:%M:%S")
# logger = logging.getLogger(__name__)
logger = get_logger("DDPTrainer")



class DDPTrainer(BaseTrainer):
    """
    分布式数据并行训练器（仅支持 torchrun 启动）。

    与 BaseTrainer 的关系：核心流程（fit / train_epoch / evaluate_epoch / test）
    和组件（优化器构建、早停、History、Visualizer、Metrics）完全复用父类，
    仅在必要处做分布式适配：

    - 进程组：构造时自动 setup_distributed()；未经 torchrun 启动时
      退化为单进程，行为与 BaseTrainer 完全一致
    - 设备：由 LOCAL_RANK 决定（CUDA → cuda:local_rank；否则 CPU + gloo，
      DDP 不支持 MPS），传入的 device 参数在分布式模式下被覆盖
    - 数据：自动用 DistributedSampler 重建 train/val/test loader（分片），
      每轮训练前 sampler.set_epoch 保证各 epoch shuffle 不同
    - 聚合：混淆矩阵类指标在 compute 前 all_reduce（全局精确）；
      train/val/test 的 loss 跨 rank 取均值
    - 落盘：仅 rank0 写 checkpoint / 日志文件 / 曲线图 / TensorBoard，
      输出目录时间戳由 rank0 广播，所有 rank 共享同一目录路径

    使用约束：
    - 启动：torchrun --nproc_per_node=N train.py；所有 rank 都要调用
      fit() / test()（内含集合通信，缺席会死锁），结束后调 cleanup()
    - val/test 集不能被 world_size 整除时，DistributedSampler 会补齐重复
      样本，全局指标有微小偏差；精确评估请单进程运行
    - 父类的 OOM 跳批容错在 DDP 下不可依赖：某 rank 跳批会造成梯度同步
      错位而挂死，请确保 batch size 在安全范围
    - save_predictions 在分片评估下只能覆盖当前 rank 的分片，已自动禁用
    """

    def __init__(self, model: nn.Module,
                 find_unused_parameters: bool = False,
                 **kwargs):
        """
        初始化分布式训练器

        Args:
            model: 待训练模型（未包装，内部自动套 DistributedDataParallel）
            find_unused_parameters: 前向中存在未参与 loss 的分支时置 True
                （有额外开销，多任务/多头模型按需开启）
            **kwargs: 其余参数与 BaseTrainer 完全一致；分布式模式下
                device 由 LOCAL_RANK 决定，传入值被忽略
        """
        # 进程组初始化（必须先于 super().__init__：其末尾的 init_settings
        # 会虚分发到本类覆写版，届时需要 rank 信息）
        self.distributed, self.rank, self.local_rank, self.world_size = \
            setup_distributed()
        self.is_main = self.rank == 0
        self.find_unused_parameters = find_unused_parameters
        # 记录原始 output_dir：同步 rank0 时间戳后需重建 save_dir
        self._output_dir = kwargs.get('output_dir', './output')

        if self.distributed:
            # DDP 设备由 local_rank 决定：nnl → 各进程绑定自己的 GPU；
            # 无 CUDA 时用 CPU + gloo（DDP 不支持 MPS 通信后端）
            ddp_device = (torch.device('cuda', self.local_rank)
                          if torch.cuda.is_available() else torch.device('cpu'))
            user_device = kwargs.get('device', 'auto')
            if str(user_device) not in ('auto', str(ddp_device)):
                logger.warning(
                    f"⚠️ DDP mode: device is determined by LOCAL_RANK "
                    f"({ddp_device}), ignoring device='{user_device}'"
                )
            kwargs['device'] = ddp_device
        else:
            logger.warning(
                "⚠️ Not launched via torchrun, DDPTrainer falls back to "
                "single-process BaseTrainer behavior"
            )

        super().__init__(model, **kwargs)


    def init_settings(self) -> None:
        """
        分布式初始化：统一输出目录 → 数据分片 → DDP 包装 →
        rank 分工（主进程走完整父类初始化，非主进程只做无落盘部分）。
        """
        if not self.distributed:
            super().init_settings()
            return

        # 1) 用 rank0 的时间戳统一所有进程的输出目录
        #   （各进程 datetime.now() 微小差异会导致目录分裂，
        #    非主进程将找不到 rank0 写的 best.pt）
        ts = [self.timestamp]
        dist.broadcast_object_list(ts, src=0)
        self.timestamp = ts[0]
        self.save_dir = Path(os.path.join(self._output_dir, self.timestamp))
        self.visualizer.save_dir = self.save_dir

        # 2) 数据分片：先于父类 init_settings，OneCycleLR 的 steps_per_epoch
        #    才能拿到分片后的正确 loader 长度
        self.train_loader = self._shard_loader(self.train_loader, shuffle=True)
        self.val_loader = self._shard_loader(self.val_loader, shuffle=False)
        self.test_loader = self._shard_loader(self.test_loader, shuffle=False)

        # 3) DDP 包装（在父类的 torch.compile 之前：官方推荐
        #    torch.compile(DDP(model)) 的包装顺序）
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if self.device.type == 'cuda' else None,
            find_unused_parameters=self.find_unused_parameters,
        )

        # 4) rank 分工
        if self.is_main:
            super().init_settings()
        else:
            # 非主进程：静默 + 无落盘，只做训练必需的初始化
            # （与父类 init_settings 的非落盘部分对齐）
            self.pbar_disable = True
            self.logger.setLevel(logging.WARNING)
            self.writer = None
            self.history = _NoopHistory()
            self.visualizer = _NoopVisualizer()
            self.init_optim_scheduler(self.optimizer_cfg, self.scheduler_cfg)
            if self.resume:
                self.load_model(self.resume, resume=True)
            if self.metrics is None:
                ignore_index = None if self.is_classification else 255
                self.metrics = Metrics(self.num_classes, ignore_index=ignore_index)
            if self.compile_model:
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    pass  # 主进程已记录 warning，此处静默回退

        # 5) 指标分布式聚合：compute 前 all_reduce 混淆矩阵，
        #    各 rank 得到一致的全局指标（best/早停决策天然同步）
        self.metrics = _DistributedMetrics(self.metrics)
        barrier()


    def _shard_loader(self, loader: Optional[DataLoader],
                      shuffle: bool) -> Optional[DataLoader]:
        """
        用 DistributedSampler 重建 DataLoader（保留原 loader 的关键配置），
        使调用方可以像 BaseTrainer 一样直接传入普通 DataLoader。
        """
        if loader is None:
            return None
        if isinstance(loader.sampler, DistributedSampler):
            return loader  # 调用方已自行分片
        if loader.batch_size is None:
            raise ValueError(
                "DataLoader with custom batch_sampler cannot be re-sharded "
                "automatically; build it with DistributedSampler yourself"
            )
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,  # 与 shuffle 互斥，随机性由 sampler 接管
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            persistent_workers=loader.persistent_workers,
        )


    def _unwrap_model(self) -> nn.Module:
        """剥离 torch.compile（_orig_mod）与 DDP（module）两层包装，取原始模型"""
        model = getattr(self.model, '_orig_mod', self.model)
        return getattr(model, 'module', model)


    def train_epoch(self) -> Dict[str, Any]:
        """训练一轮：set_epoch 保证分片 shuffle 逐轮不同，loss 跨 rank 聚合"""
        if self.distributed:
            # 不 set_epoch 则每个 epoch 的分片 shuffle 相同，等于没有 shuffle
            self.train_loader.sampler.set_epoch(self.current_epoch)
        results = super().train_epoch()
        if self.distributed:
            # 各 rank 分片大小相同（DistributedSampler 补齐），直接平均即可；
            # 同步修正内存曲线列表，rank0 绘图/History 用全局值
            results['loss'] = reduce_value(results['loss'], average=True)
            if self.train_loss_all:
                self.train_loss_all[-1] = results['loss']
        return results


    def evaluate_epoch(self) -> Dict[str, Any]:
        """验证一轮：acc 等指标已由 _DistributedMetrics 全局聚合，此处聚合 loss"""
        results = super().evaluate_epoch()
        if self.distributed:
            results['loss'] = reduce_value(results['loss'], average=True)
            if self.val_loss_all:
                self.val_loss_all[-1] = results['loss']
        return results


    def save_model(self, filename: str,
                   checkpoint: Optional[Dict[str, Any]] = None) -> str:
        """仅 rank0 落盘；写完后 barrier，保证其他 rank 后续可安全读取"""
        if not self.distributed:
            return super().save_model(filename, checkpoint)
        if self.is_main:
            path = super().save_model(filename, checkpoint)
        else:
            path = str(self.save_dir / filename)
        barrier()
        return path


    def _restore_history(self, src_log: Path) -> None:
        """训练历史曲线仅由 rank0 维护（非主进程的 History 为替身，无需恢复）"""
        if not self.distributed or self.is_main:
            super()._restore_history(src_log)


    @torch.no_grad()
    def test(self,
             report_results: bool = True,
             save_error_index: bool = False,
             save_predictions: bool = False) -> Optional[Dict[str, Any]]:
        """
        分片测试：所有 rank 必须同时调用（混淆矩阵聚合含集合通信）。
        报告/图表仅 rank0 输出；返回的 acc 等指标为全局值。
        """
        if self.distributed:
            if save_predictions:
                # 分片评估下各 rank 只持有自己分片的预测，索引也是分片局部的，
                # 导出结果不完整且有误导性 → 禁用（需要完整预测请单进程跑 test）
                self.logger.warning(
                    "⚠️ save_predictions disabled under DDP sharded evaluation "
                    "(each rank only sees its own shard); run test() in "
                    "single-process mode to export full predictions"
                )
                save_predictions = False
                save_error_index = False
            if not self.is_main:
                report_results = False  # 报告只由 rank0 打印

        results = super().test(
            report_results=report_results,
            save_error_index=save_error_index,
            save_predictions=save_predictions,
        )

        if self.distributed and results is not None:
            results['loss'] = reduce_value(results['loss'], average=True)
            results['samples'] = int(reduce_value(float(results['samples']),
                                                  average=False))
        return results


    def cleanup(self) -> None:
        """销毁进程组（脚本结束前调用；内含 barrier，确保各 rank 都完成）"""
        cleanup_distributed()

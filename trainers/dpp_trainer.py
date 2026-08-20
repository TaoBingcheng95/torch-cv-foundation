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
from .tb_logger import TensorBoardLogger

from .basetrainer import BaseTrainer
from .ddp_utils import (
    setup_distributed, cleanup_distributed, barrier,
    reduce_value, _NoopHistory, _NoopVisualizer, _DistributedMetrics
)

from metrics import ClassificationMetric, SegmentationMetric
from utils.logger import get_logger

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
    和组件（优化器构建、早停、History、Visualizer、指标计算器）完全复用父类，
    仅在必要处做分布式适配：

    - 进程组：构造时自动 setup_distributed()；未经 torchrun 启动时
      退化为单进程，行为与 BaseTrainer 完全一致
    - 模型：构造时通过 _wrap_model 钩子套 DistributedDataParallel（先于 .to(device)），
      torch.compile 在父类 init_settings 中再包一层，形成 compile(DDP(model)) —— 官方推荐顺序
    - 设备：由 LOCAL_RANK 决定（CUDA → cuda:local_rank；否则 CPU + gloo，
      DDP 不支持 MPS），传入的 device 参数在分布式模式下被覆盖
    - 数据：自动用 DistributedSampler 重建 train/val/test loader（分片），
      每轮训练前 sampler.set_epoch 保证各 epoch shuffle 不同
    - 损失聚合：覆写 _aggregate_loss 钩子，在 epoch 末尾 ``if total_samples==0``
      检查前跨 rank all_reduce(SUM) total_loss 与 total_samples；
      下游 avg_loss 天然得到全局加权均值，OOM 全空时各 rank 同步看到 0 信号避免死锁
    - 指标聚合：原生 PyTorch 指标类（ClassificationMetric / SegmentationMetric）
      在 compute 前显式 all_reduce 混淆矩阵（全局精确）；
      torchmetrics 指标无 all_reduce()，其 compute() 自动同步 DDP
    - 落盘：仅 rank0 写 checkpoint / 日志文件 / 曲线图 / TensorBoard，
      输出目录时间戳由 rank0 广播，所有 rank 共享同一目录路径

    使用约束：
    - 启动：torchrun --nproc_per_node=N train.py；所有 rank 都要调用
      fit() / test()（内含集合通信，缺席会死锁），结束后调 cleanup()
      或用 ``with DDPTrainer(...) as trainer:`` 上下文管理器自动清理
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

        # 切换到 DDPTrainer 自己的 logger（分布式模式下区分日志来源）
        # 必须先于 super().init_settings()：父类据此清理旧 FileHandler 并挂新 FileHandler
        self.logger = logger
        self.visualizer.logger = logger

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

        # 3) DDP 包装已在 __init__ 的 _wrap_model 钩子中完成（先于 .to(device)）；
        #    torch.compile 在父类 init_settings 中对 DDP(model) 再包装，
        #    形成 compile(DDP(model)) —— 官方推荐的包装顺序

        # 4) rank 分工
        if self.is_main:
            super().init_settings()
        else:
            # 非主进程：静默 + 无落盘，只做训练必需的初始化
            # （与父类 init_settings 的非落盘部分对齐）
            self.pbar_disable = True
            self.logger.setLevel(logging.WARNING)
            self.tb_logger = TensorBoardLogger(log_dir='', enabled=False)
            self.history = _NoopHistory()
            self.visualizer = _NoopVisualizer()
            self._detect_scheduler_type()
            if self.resume:
                self.load_checkpoint(self.resume, resume=True)
            if self.metrics is None:
                # 与 BaseTrainer.init_settings 同步：按任务类型选择计算器
                self.metrics = (ClassificationMetric(self.num_classes)
                                if self.is_classification
                                else SegmentationMetric(self.num_classes))
            if self.compile_model:
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    pass  # 主进程已记录 warning，此处静默回退

        # 5) 指标分布式聚合：原生 PyTorch 指标类自带 all_reduce()，
        #    _DistributedMetrics 包装器在 compute/per_class_metrics 前自动调用；
        #    state 必须在通信设备上（NCCL 要求 CUDA tensor），
        #    否则 all_reduce 会在 CPU 上调用集合通信而报错；
        #    主进程已在 super().init_settings() 内 .to，此处覆盖非主进程分支
        self.metrics = self.metrics.to(self.device)
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


    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """套 DistributedDataParallel；非分布式模式原样返回（退化为 BaseTrainer）。

        DDP 要求模型已在目标设备上（device_ids 指向的 GPU），故先 .to(device)
        再包装；基类 __init__ 随后对返回值 .to(device) 是 no-op，不会重复搬运。
        CPU(gloo) 下不传 device_ids（NCCL 才需要）。
        """
        if not self.distributed:
            return model
        model = model.to(self.device)
        kwargs = dict(find_unused_parameters=self.find_unused_parameters)
        if self.device.type == 'cuda':
            kwargs['device_ids'] = [self.local_rank]
        return DistributedDataParallel(model, **kwargs)


    def _aggregate_loss(self, total_loss, total_samples):
        """跨 rank all_reduce(SUM) total_loss 张量与 total_samples 标量。

        覆写基类 no-op 钩子：训练/验证/测试 epoch 末尾在 ``if total_samples == 0``
        检查前先做一次跨 rank 同步，所有 rank 完成后返回同一
        (global_loss, global_samples)：

        - 损失指标计算口径一致：下游 ``avg_loss = total_loss / total_samples``
          得到全局加权均值，train/loss、val/loss、test/loss 各 rank 完全一致
        - OOM 全空时跨 rank 同步：某 rank 数据量极少全空抛异常退出时，其他
          rank 也能通过 all_reduce 收到 total_samples=0 信号一起抛，避免
          部分进程已退出、其他进程卡在 ``metric.compute()`` 的 all_reduce 上死锁
          （指标同步依赖 torchmetrics 内置 sync，见类 docstring）

        非分布式模式（torchrun 未启动，DDPTrainer 退化为单进程）走基类 no-op。
        """
        if not self.distributed:
            return super()._aggregate_loss(total_loss, total_samples)
        # total_loss: 张量，all_reduce(SUM) 返回张量；
        # total_samples: int → 转 float reduce 后再转回 int
        global_loss = reduce_value(total_loss, average=False)
        global_samples = int(reduce_value(total_samples, average=False))
        return global_loss, global_samples


    def train_epoch(self) -> Dict[str, Any]:
        """训练一轮：set_epoch 保证分片 shuffle 逐轮不同；loss 由 _aggregate_loss 跨 rank 聚合"""
        if self.distributed:
            # 不 set_epoch 则每个 epoch 的分片 shuffle 相同，等于没有 shuffle
            self.train_loader.sampler.set_epoch(self.current_epoch)
        # 父类 train_epoch 在 if total_samples==0 检查前调 _aggregate_loss，
        # 已将 total_loss / total_samples 跨 rank all_reduce(SUM)；
        # 下游 avg_loss = total_loss / total_samples 自然得到全局加权均值，
        # train_loss_all.append 的也是全局值，rank0 绘图/History 无需后处理
        return super().train_epoch()


    def evaluate_epoch(self) -> Dict[str, Any]:
        """验证一轮：先 all_reduce 混淆矩阵，再让父类 compute 全局指标；loss 由 _aggregate_loss 聚合"""
        if self.distributed and hasattr(self._val_metric, 'all_reduce'):
            # 父类 evaluate_epoch 内部 self.metrics = self._val_metric 会绕过
            # _DistributedMetrics 包装器，因此需在此显式 all_reduce；
            # reset() 后首次 compute 前调用一次即可，all_reduce 原地修改矩阵
            # torchmetrics 指标无 all_reduce()，其 compute() 自动同步 DDP
            self._val_metric.all_reduce()
        # 父类 evaluate_epoch 在 if total_samples==0 检查前调 _aggregate_loss，
        # 已将 total_loss / total_samples 跨 rank all_reduce(SUM)；
        # val_loss_all.append 的也是全局值，rank0 绘图/hparam 无需后处理
        return super().evaluate_epoch()


    def save_checkpoint(self, filename: str,
                        checkpoint: Optional[Dict[str, Any]] = None) -> str:
        """仅 rank0 落盘；写完后 barrier，保证其他 rank 后续可安全读取。

        覆写基类 save_checkpoint：基类 fit() 在每个 epoch 末调
        ``self.save_checkpoint('best.pt'/'last.pt', ...)``，若不门控则所有 rank
        并发写同一文件会互相覆盖（best.pt 可能保存到非最优 rank 的权重）。
        """
        if not self.distributed:
            return super().save_checkpoint(filename, checkpoint)
        if self.is_main:
            path = super().save_checkpoint(filename, checkpoint)
        else:
            path = str(self.save_dir / filename)
        barrier()
        return path


    def _restore_history(self, src_log: Path) -> None:
        """训练历史曲线仅由 rank0 维护（非主进程的 History 为替身，无需恢复）"""
        if not self.distributed or self.is_main:
            super()._restore_history(src_log)

    def _restore_tensorboard(self, src_tb_dir: Path) -> None:
        """TensorBoard 事件文件迁移仅由 rank0 执行（非主进程的 tb_logger 为禁用实例）"""
        if not self.distributed or self.is_main:
            super()._restore_tensorboard(src_tb_dir)


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
            # 父类 test 内部 self.metrics = self._test_metric 会绕过
            # _DistributedMetrics 包装器，因此需在此显式 all_reduce
            # torchmetrics 指标无 all_reduce()，其 compute() 自动同步 DDP
            if hasattr(self._test_metric, 'all_reduce'):
                self._test_metric.all_reduce()

        results = super().test(
            report_results=report_results,
            save_error_index=save_error_index,
            save_predictions=save_predictions,
        )
        # 父类 test 在 if total_samples==0 检查前调 _aggregate_loss，
        # 已将 total_loss / total_samples 跨 rank all_reduce(SUM)；
        # results['loss']、results['samples'] 已是全局值，无需后处理。
        # 混淆矩阵在 basetrainer.test() 中通过 .confusion_matrix 属性获取，
        # 内部已 all_reduce，亦无需重复计算
        return results


    def cleanup(self) -> None:
        """销毁进程组（脚本结束前调用；内含 barrier，确保各 rank 都完成）

        幂等：多次调用安全（用 _cleaned_up 标记位防止重复销毁）；
        非分布式模式直接返回（未初始化进程组，无可销毁资源）。
        """
        if getattr(self, '_cleaned_up', False):
            return
        if not self.distributed:
            # 单进程退化模式：仅清理父类资源（history 已在 fit 末尾关闭，
            # 此处确保即使异常退出也释放 persistent workers）
            super().cleanup() if hasattr(super(), 'cleanup') else None
            self._cleaned_up = True
            return
        # 分布式模式：barrier 保证各 rank 都完成后续操作（避免某 rank
        # 退出后其他 rank 卡在未完成的集合通信上），再销毁进程组
        cleanup_distributed()
        self._cleaned_up = True

    def __enter__(self):
        """支持 ``with DDPTrainer(...) as trainer:`` 用法，退出时自动 cleanup"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """异常退出时也确保进程组销毁，避免僵尸进程或 NCCL 句柄泄漏"""
        self.cleanup()
        return False  # 不吞异常，让上层感知

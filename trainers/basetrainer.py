
import os
import re
from pathlib import Path
import time
from datetime import datetime
import json
from typing import Dict, Optional, Any, List, Tuple, Union

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 混合精度训练
from torch.amp import autocast
from torch.amp import GradScaler

from .visualizer import TrainingVisualizer
from .utils import EarlyStopping, History
from .logger import get_logger, add_file_handler

from metrics import Metrics
from optimizers import build_optimizer, build_scheduler, clip_grad_norm
from utils.hardware import select_device, collect_hardware_report

# 日志配置
# import logging
# logging.basicConfig(level=logging.INFO, 
#                     format="%(asctime)s - %(levelname)s - %(message)s",
#                     datefmt="%Y-%m-%d %H:%M:%S")
# logger = logging.getLogger(__name__)
logger = get_logger("BaseTrainer")


class BaseTrainer:
    """
    通用深度学习训练器基类，支持分类和分割任务
    
    核心功能:
        - 自动设备检测与分配
        - 灵活的优化器/调度器配置
        - 完整的训练/验证/测试流程
        - 早停机制和模型检查点
        - 丰富的日志和可视化输出
    """
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 class_names: Optional[List[str]] = None,
                 is_classification: bool = True, # 是否为分类任务（影响指标计算和日志记录）
                 num_classes: int = 10,
                 epochs: int = 30,
                 log_interval: int = 5,
                 eval_interval: int = 5,  # 每隔多少个 epoch 验证一次（1 = 每轮都验证）
                 device: Union[str, torch.device] = 'auto',  # 'auto' | 'cuda' | 'cpu' | 'mps'，也可直接传 torch.device
                 optimizer_cfg: Optional[Dict[str, Any]] = None,
                 scheduler_cfg: Optional[Dict[str, Any]] = None,
                 criterion: Optional[nn.Module] = None,  # None 时默认 CrossEntropyLoss
                 metrics = None,
                 resume: Optional[str]=None,
                 compile_model:bool = False, 
                 use_amp: bool = False,  # 混合精度训练（AMP）
                 max_grad_norm: Optional[float] = None,  # 梯度裁剪
                 early_stop_patience: Optional[int] = 10,  # 早停容忍次数（None 表示禁用早停）
                 early_stop_delta: float = 0.0,  # 早停判定的最小改善阈值
                 monitor: str = 'loss',  # 统一监控指标：best.pt / 早停 / Plateau 共用
                 monitor_mode: str = 'auto',  # 'auto' | 'min' | 'max'
                 output_dir: str='./output',
                 use_tensorboard: bool = True,):  # **kwargs     
        """
        初始化训练器
        
        :param optimizer_cfg: 优化器配置字典
            示例: {"type": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "momentum": 0.9}
        :param scheduler_cfg: 调度器配置字典（None 表示不使用）
            示例: {"type": "reduceLROnPlateau", "mode": "min", "patience": 5, "factor": 0.5}
        :param early_stop_patience: 早停容忍次数，监控指标连续多少次不改善后停训；
            None 表示禁用早停。注意：
            - eval_interval > 1 时按“验证次数”而非 epoch 数计；
            - 搭配 ReduceLROnPlateau 时应大于其 patience（建议 2~3 倍），
              否则 LR 还没来得及衰减就会触发早停。
        :param monitor: 统一监控指标，best.pt 保存 / 早停 / ReduceLROnPlateau 共用。
            可选 'loss'、'acc'，或 Metrics.compute() 结果中的任意键（如 'oa'、'miou'）
        :param monitor_mode: 'auto' 按名称推断（名称含 'loss' → min，其余 → max），
            也可显式指定 'min' / 'max'
        :param use_tensorboard: 是否启用 TensorBoard 标准日志。writer 在
            init_settings 中创建（写入 save_dir/tensorboard），生命周期由
            trainer 全程管理，fit() 结束时自动关闭
        :param use_amp: 是否启用混合精度训练。按设备自适应：
            - CUDA: float16 autocast + GradScaler（损失缩放防梯度下溢）
            - CPU : bfloat16 autocast（无需缩放）
            - MPS : float16 autocast（无 scaler，极端小损失时注意下溢风险）
        """
        # 时间戳（用于输出目录命名）
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设备配置（统一委托 select_device：auto 按 cuda → mps → cpu 选择，
        # 显式指定不可用后端时 fail fast；入参幂等，调用方预先解析过也可直接传入）
        self.device = select_device(device)

        # 输出目录
        self.save_dir = Path(os.path.join(output_dir, self.timestamp))

        # 核心组件
        self.model = model.to(self.device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class-{i}' for i in range(num_classes)]

        # 任务配置
        self.epochs = epochs
        # 当前 epoch
        self.current_epoch = 0
        # 起始 epoch（断点续训时由 load_model 恢复）
        self.start_epoch = 0
        # 统一监控指标：best.pt 保存、早停、ReduceLROnPlateau 共用同一 monitor
        self.monitor = monitor
        if monitor_mode == 'auto':
            self.monitor_mode = 'min' if 'loss' in monitor else 'max'
        elif monitor_mode in ('min', 'max'):
            self.monitor_mode = monitor_mode
        else:
            raise ValueError(
                f"monitor_mode must be 'auto'/'min'/'max', got '{monitor_mode}'")
        # 历史最佳监控值（断点续训时由 load_model 恢复，避免 best.pt 被更差模型覆盖）
        self.best_metric = float('inf') if self.monitor_mode == 'min' else float('-inf')
        # 全局步数（batch 级 TensorBoard 曲线的横轴，断点续训时在 fit 中推算衔接）
        self.global_step = 0
        self.log_interval = log_interval
        self.eval_interval = max(1, eval_interval)
        self.is_classification = is_classification

        # 损失函数（默认在此实例化，避免可变默认参数被多个实例共享；
        # 显式 to(device)：带 weight 等 buffer 的 loss 需与模型同设备）
        self.criterion = (criterion or nn.CrossEntropyLoss()).to(self.device)
        # 优化器与调度器配置
        self.optimizer = None
        self.scheduler = None
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.max_grad_norm = max_grad_norm
        # 早停配置（与调度器配置解耦，patience=None 表示禁用早停）
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        # 是否为 batch 级调度器（如 OneCycleLR），在 init_optim_scheduler 中按实例类型判定
        self.is_batch_scheduler = False

        # 恢复训练
        self.resume = resume
        # 编译选项
        self.compile_model = compile_model
        # 混合精度训练（AMP）：autocast 精度按设备选择，
        # GradScaler 仅 CUDA + fp16 需要；其余设备构造 disabled 的 scaler，
        # 其 scale/step/update 自动退化为透传，训练循环无需分支
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if self.device.type == 'cpu' else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == 'cuda')
        # 日志器
        self.logger = logger
        # 进度条开关（DDP 非主进程置 True 避免多进程进度条交错刷屏）
        self.pbar_disable = False

        # TensorBoard writer（标准日志开关；writer 在 init_settings 中创建，
        # 因 save_dir 需先建目录；DDP 非主进程不走 init_settings，天然不创建）
        self.use_tensorboard = use_tensorboard
        self.writer = None

        # 可视化器：只持有展示配置（输出目录/类别名），训练数据由调用时显式传入
        self.visualizer = TrainingVisualizer(
            save_dir=self.save_dir,
            class_names=self.class_names,
            logger=self.logger,
        )

        # 指标记录（内存列表供绘图；持久化由 History 组件写入 JSONL，
        # 断点续训时由 load_model 从旧目录的日志恢复，保证曲线完整衔接；
        # 全新训练应新建 Trainer 实例）
        self.metrics = metrics
        self.train_loss_all = []
        self.val_loss_all = []
        self.val_acc_all = []
        self.val_epochs = []  # 记录每次验证对应的 epoch（eval_interval > 1 时绘图用）
        self.lr_history = []
        self.cnf_matrix = None
        self.val_metrics_result = None

        self.init_settings()


    def init_settings(self) -> None:
        """初始化训练环境"""
        self.logger.info("📋 Initializing training environment...")

        # 输出目录
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"📁 Output directory: {self.save_dir}")
        add_file_handler(self.logger, self.save_dir/ "train.log")

        # 训练历史记录器（JSONL 逐条追加 + flush，崩溃安全；
        self.history = History(self.save_dir / 'training_log.jsonl')

        # TensorBoard writer（标准日志，与 checkpoint 同目录便于归档对比；
        # 查看：tensorboard --logdir <output_dir>，各时间戳子目录自动识别为 run）
        if self.use_tensorboard:
            tb_dir = self.save_dir / 'tensorboard'
            self.writer = SummaryWriter(log_dir=str(tb_dir))
            self.logger.info(f"📈 TensorBoard logging to: {tb_dir}")

        self.logger.info(f"🤖 Setting up device: {self.device}")

        # 硬件环境快照（复现实验用，采集失败不阻断训练）
        try:
            report_path = self.save_dir / "hardware.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(collect_hardware_report(), f, indent=2, sort_keys=True)
            self.logger.info(f"💻 Hardware report saved: {report_path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save hardware report: {e}")

        # 混合精度状态
        if self.use_amp:
            self.logger.info(
                f"⚡ AMP enabled | autocast dtype: {self.amp_dtype} | "
                f"GradScaler: {'on' if self.scaler.is_enabled() else 'off (non-CUDA)'}"
            )

        # 优化器和调度器
        self.logger.info("🔧 Initializing optimizer and scheduler...")
        self.logger.info(f"📌 Monitor: {self.monitor} (mode: {self.monitor_mode})")
        self.init_optim_scheduler(self.optimizer_cfg, self.scheduler_cfg)

        # 恢复训练（resume=True: 同时恢复 epoch/best_metric/优化器/调度器状态）
        if self.resume:
            self.logger.info(f"📥 Resuming from checkpoint: {self.resume}")
            self.load_model(self.resume, resume=True)

        # 指标计算器（基于混淆矩阵，在 CPU 上累积，避免 GPU 内存占用过高）
        self.logger.info("📊 Initializing metrics calculator...")
        if self.metrics is None:
            # 分类任务不忽略任何标签；分割任务默认忽略 255（VOC 等数据集
            # 未标注区域的通用约定）；如需其他 ignore_index，直接传入自定义 metrics 实例
            ignore_index = None if self.is_classification else 255
            self.metrics = Metrics(self.num_classes, ignore_index=ignore_index)

        # 模型编译（PyTorch 2.0+）
        if self.compile_model:
            try:
                self.logger.info("Compiling model with torch.compile...")
                self.model = torch.compile(self.model)
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}, using original model")
        self.logger.info("✅ Initialization complete!")


    def init_optim_scheduler(
            self,
            optimizer_cfg: Optional[Dict[str, Any]] = None,
            scheduler_cfg: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化优化器和学习率调度器（委托 optimizers.builder 统一构建）
        :param optimizer_cfg: 优化器配置字典，字段说明见 build_optimizer
        :param scheduler_cfg: 调度器配置字典，字段说明见 build_scheduler（None 表示固定学习率）
        """
        # ========== 优化器 ==========
        self.optimizer = build_optimizer(self.model, optimizer_cfg)
        self.logger.info(
            f"🎯 Optimizer: {type(self.optimizer).__name__} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
            f"Weight Decay: {self.optimizer.param_groups[0]['weight_decay']:.2e} | "
            f"Param Groups: {len(self.optimizer.param_groups)}"
        )

        # ========== 调度器 ==========
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_cfg,
            total_epochs=self.epochs,
            steps_per_epoch=len(self.train_loader) if self.train_loader else None,
        )
        if self.scheduler is None:
            self.logger.info("Scheduler: None (using constant learning rate)")
        else:
            self.logger.info(f"Scheduler: {type(self.scheduler).__name__}")
        # OneCycleLR 在每个 batch 后 step，其余调度器在每个 epoch 后 step
        self.is_batch_scheduler = isinstance(self.scheduler, lr_scheduler.OneCycleLR)

        # Plateau 与统一 monitor 的 mode 对齐检查（不一致时 LR 衰减方向会反）
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau) and \
                self.scheduler.mode != self.monitor_mode:
            self.logger.warning(
                f"⚠️ ReduceLROnPlateau mode '{self.scheduler.mode}' != monitor_mode "
                f"'{self.monitor_mode}' (monitor='{self.monitor}'), "
                f"set scheduler_cfg['mode'] = '{self.monitor_mode}' to align"
            )


    def _empty_cache(self) -> None:
        """
        按设备类型释放缓存分配器占用的显存（OOM 恢复用）。

        CUDA / MPS 分别调用各自后端的 empty_cache；CPU 无需操作。
        注意：MPS 的 OOM 报错同样包含 "out of memory"，若硬编码
        torch.cuda.empty_cache() 会静默无效，缓存得不到释放。
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.empty_cache()


    def _is_better(self, value: float) -> bool:
        """判断监控指标是否优于历史最佳（按 monitor_mode 方向比较）"""
        if self.monitor_mode == 'min':
            return value < self.best_metric
        return value > self.best_metric


    def _step_scheduler(self, val_metrics: Dict[str, float]) -> float:
        """
        统一处理调度器 step，返回当前学习率
        
        Args:
            val_metrics: 验证集指标字典
        
        Returns:
            当前学习率
        """
        if self.scheduler is None:
            return self.optimizer.param_groups[0]['lr']
        
        # OneCycleLR 在 train_epoch 内按 batch 已调用
        if self.is_batch_scheduler:
            return self.optimizer.param_groups[0]['lr']
        # 区分调度器类型
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            # 本轮未验证：Plateau 依赖验证指标，跳过 step（其余调度器不受影响）
            if val_metrics is None:
                return self.optimizer.param_groups[0]['lr']
            # 统一监控指标（与 best.pt / 早停一致），
            # mode 对齐已在 init_optim_scheduler 构建时校验
            metric = val_metrics.get(self.monitor)
            if metric is None:
                self.logger.warning(
                    f"Monitor key '{self.monitor}' not found in val_metrics, using 'loss'"
                )
                metric = val_metrics['loss']
            
            self.scheduler.step(metric)
            self.logger.debug(
                f"ReduceLROnPlateau step: {self.monitor}={metric:.4f}"
            )
        else:
            self.scheduler.step()
        
        return self.optimizer.param_groups[0]['lr']


    # ==================== SummaryWriter 标准日志接口层 ====================
    # 两套日志分工：自定义日志（logger / History JSONL / visualizer）面向
    # 实验过程中的快速查看；以下方法统一收拢 SummaryWriter 调用，按业内
    # 标准工具的形式记录（标量曲线 / 模型结构 / 超参对比）。
    # 后续迁移 wandb / mlflow 等工具时，只需替换这一层实现，
    # 训练主流程（fit / train_epoch / evaluate_epoch）无需改动。

    def log_scalars(self,
                    scalars: Dict[str, Any],
                    step: int,
                    prefix: str = '') -> None:
        """
        批量记录标量曲线。writer 未启用时静默跳过；非数值项自动忽略。

        Args:
            scalars: 标量字典，如 {'epoch_loss': 0.5, 'learning_rate': 1e-3}
            step: 横轴步数（epoch 级传 current_epoch，batch 级传 global_step）
            prefix: 标签前缀（如 'train/'、'val/'，用于面板分组）
        """
        if self.writer is None:
            return
        for name, value in scalars.items():
            # bool 是 int 子类但无曲线意义；time 等非数值项直接跳过
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.writer.add_scalar(f'{prefix}{name}', value, step)


    def log_val_metrics(self, results: Dict[str, float], step: int) -> None:
        """
        记录 Metrics.compute() 的全部验证指标。

        汇总指标（oa/mpa/miou/fwiou/mf1 等）写入 'val/'；
        逐类指标（iou_0/precision_1 等）按类别名分组写入 'val_per_class/'，
        避免类别较多时污染主面板。
        """
        if self.writer is None:
            return
        summary, per_class = {}, {}
        for key, value in results.items():
            m = re.match(r'^(.+)_(\d+)$', key)
            if m and int(m.group(2)) < self.num_classes:
                base, idx = m.group(1), int(m.group(2))
                per_class[f'{base}/{self.class_names[idx]}'] = value
            else:
                summary[key] = value
        self.log_scalars(summary, step, 'val/')
        self.log_scalars(per_class, step, 'val_per_class/')


    def log_graph(self) -> None:
        """
        记录模型结构图（训练开始时调用一次；失败仅告警不阻断训练）。

        add_graph 基于 trace，部分动态控制流模型 / 特殊设备可能失败，
        因此包在 try 中；trace 对象用 _unwrap_model 避开 torch.compile 包装层。
        """
        if self.writer is None or self.train_loader is None:
            return
        try:
            sample, _ = next(iter(self.train_loader))
            self.writer.add_graph(self._unwrap_model(), sample.to(self.device))
            self.logger.info("📈 Model graph logged to TensorBoard")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log model graph: {e}")


    def log_hparams(self, metrics: Dict[str, float]) -> None:
        """
        记录超参数与最终指标的对照表（训练结束时调用一次，
        供 TensorBoard HPARAMS 面板跨实验对比）。

        Args:
            metrics: 最终指标字典，键名建议带 'hparam/' 前缀（业内惯例，
                     与普通标量曲线区分）
        """
        if self.writer is None:
            return
        # 基础超参（add_hparams 仅支持 int/float/str/bool，其余类型转 str）
        hparams: Dict[str, Any] = {
            'model': type(self._unwrap_model()).__name__,
            'criterion': type(self.criterion).__name__,
            'epochs': self.epochs,
            'use_amp': self.use_amp,
            'monitor': self.monitor,
        }
        batch_size = getattr(self.train_loader, 'batch_size', None)
        if batch_size is not None:
            hparams['batch_size'] = batch_size
        if self.max_grad_norm is not None:
            hparams['max_grad_norm'] = self.max_grad_norm
        # 展平优化器/调度器配置（嵌套字典不被 add_hparams 支持）
        for cfg_name, cfg in (('optim', self.optimizer_cfg),
                              ('sched', self.scheduler_cfg)):
            for k, v in (cfg or {}).items():
                hparams[f'{cfg_name}/{k}'] = (
                    v if isinstance(v, (int, float, str, bool)) else str(v))
        try:
            # run_name='.' 写入当前 run 目录，避免产生额外的时间戳子 run
            self.writer.add_hparams(hparams, metrics, run_name='.')
            self.logger.info("📈 Hparams logged to TensorBoard")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log hparams: {e}")


    def fit(self) -> None:
        """
        执行完整训练流程。

        每个 epoch：train_epoch() 训练 → 按 eval_interval 调用 evaluate_epoch() 验证
        （最后一轮强制验证）→ 调度器 step → 训练历史落盘（JSONL）→
        monitor 改善时保存 best.pt → 早停判断 → 保存 last.pt。
        训练结束后自动恢复 best.pt 权重并绘制训练曲线；
        测试与训练解耦，由用户在 fit() 结束后手动调用 test()。
        """

        self.logger.info("🚀 Starting training...")
        
        # 调度器信息
        if self.scheduler is None:
            self.logger.info("📋 Scheduler: None (fixed learning rate)")
        else:
            sched_name = type(self.scheduler).__name__
            self.logger.info(f"📋 Scheduler: {sched_name}")
            if sched_name == "ReduceLROnPlateau":
                self.logger.info(
                    f"   • mode: {self.scheduler.mode}, "
                    f"patience: {self.scheduler.patience}, "
                    f"factor: {self.scheduler.factor}"
                )
            elif sched_name == "StepLR":
                self.logger.info(
                    f"   • step_size: {self.scheduler.step_size}, "
                    f"gamma: {self.scheduler.gamma}"
                )
        
        # 初始学习率
        init_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"🎯 Initial LR: {init_lr:.2e}")

        # 早停器（与 best.pt / Plateau 共用统一 monitor，改善方向由 monitor_mode 决定）
        early_stopper = None
        if self.early_stop_patience is not None:
            early_stopper = EarlyStopping(
                patience=self.early_stop_patience,
                delta=self.early_stop_delta,
                mode=self.monitor_mode,
                verbose=False,
            )
            self.logger.info(
                f"⏳ Early stopping enabled | monitor: {self.monitor} ({self.monitor_mode}) | "
                f"patience: {self.early_stop_patience} "
                f"validation rounds | delta: {self.early_stop_delta}"
            )
            # 搭配 Plateau 调度器时，早停 patience 应大于降 LR 的 patience，
            # 否则 LR 还没来得及衰减就触发早停，Plateau 形同虚设
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                plateau_patience = self.scheduler.patience
                if self.early_stop_patience <= plateau_patience:
                    self.logger.warning(
                        f"⚠️ early_stop_patience ({self.early_stop_patience}) <= "
                        f"ReduceLROnPlateau patience ({plateau_patience}), "
                        f"LR decay may never take effect before early stopping. "
                        f"Recommended: early_stop_patience >= {plateau_patience * 2}"
                    )
        else:
            self.logger.info("⏳ Early stopping disabled")

        if self.start_epoch > 0:
            self.logger.info(
                f"🔄 Resuming training from epoch {self.start_epoch + 1} "
                f"(best {self.monitor} so far: {self.best_metric:.4f})"
            )
            # batch 级曲线横轴衔接（按已完成 epoch 数推算，避免续训后曲线重叠）
            if self.train_loader is not None:
                self.global_step = self.start_epoch * len(self.train_loader)
        if self.start_epoch >= self.epochs:
            self.logger.warning(
                f"⚠️ start_epoch ({self.start_epoch}) >= epochs ({self.epochs}), "
                f"skipping training loop"
            )

        # 标准日志：训练开始时记录一次模型结构图
        self.log_graph()

        for epoch in range(self.start_epoch, self.epochs):

            self.current_epoch = epoch + 1
            self.logger.info(f"📅 Epoch {self.current_epoch} / {self.epochs}")
            
            # 训练
            train_results = self.train_epoch()

            # 验证：每 eval_interval 轮一次；最后一轮强制验证，确保 best.pt 能覆盖末期模型
            should_validate = self.val_loader is not None and (
                self.current_epoch % self.eval_interval == 0
                or self.current_epoch == self.epochs
            )
            if should_validate:
                val_metrics = self.evaluate_epoch()
                # 统一 monitor 必须存在于验证指标中（首轮验证即 fail fast）
                if self.monitor not in val_metrics:
                    raise KeyError(
                        f"monitor '{self.monitor}' not found in validation metrics, "
                        f"available keys: {sorted(val_metrics.keys())}"
                    )
            else:
                val_metrics = None
                # if self.val_loader is None:
                #     self.logger.warning("⚠️ No validation loader, skipping validation")

            # 调整学习率（Plateau 类调度器仅在有验证结果的轮次 step）
            current_lr = self._step_scheduler(val_metrics)
            self.lr_history.append(current_lr)

            # 训练历史落盘（JSONL 逐条追加，断点续训时据此衔接曲线）
            self.history.append({
                'phase': 'train', 'epoch': self.current_epoch,
                'loss': train_results['loss'], 'lr': current_lr,
                'time': train_results['time'],
            })
            if val_metrics is not None:
                self.history.append({
                    'phase': 'val', 'epoch': self.current_epoch, **val_metrics,
                })

            # ========== ✅ 验证轮次专属：更新最佳模型 + 早停判断 ==========
            # （先于 last.pt 保存，保证 last.pt 记录的 best_metric 是最新值）
            should_stop = False
            if val_metrics is not None:
                monitored = val_metrics[self.monitor]
                if self._is_better(monitored):
                    self.best_metric = monitored
                    best_checkpoint = {
                        'monitor': self.monitor,
                        'best_metric': self.best_metric,
                        'val_acc': val_metrics['acc'],
                        'val_loss': val_metrics['loss'],
                        'epoch': self.current_epoch,
                        'model': self._unwrap_model().state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                        'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
                        'config': {  # ✅ 额外保存配置，方便复现
                            'optimizer_cfg': self.optimizer_cfg,
                            'scheduler_cfg': self.scheduler_cfg,
                        }
                    }

                    # 保存固定文件名 best.pt
                    self.save_model('best.pt', checkpoint=best_checkpoint)

                    self.logger.info(
                        f"✨ New best model saved! | "
                        f"Epoch: {self.current_epoch} | "
                        f"{self.monitor}: {monitored:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f}"
                    )

                # 早停检查（监控与 best.pt 相同的 monitor；仅判断是否继续训练，
                # eval_interval > 1 时 patience 按“验证次数”而非 epoch 数计）
                if early_stopper is not None:
                    early_stopper(
                        value=monitored, 
                        epoch=self.current_epoch)
                    if early_stopper.early_stop:
                        self.logger.info(
                            f"🛑 Early stopping triggered at epoch {self.current_epoch} "
                            f"(no {self.monitor} improvement for "
                            f"{early_stopper.patience} validation rounds)"
                        )
                        should_stop = True

            # ========== ✅ 保存最新模型 (last.pt) ==========
            last_checkpoint = {
                'epoch': self.current_epoch,
                'model': self._unwrap_model().state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
                'val_loss': val_metrics['loss'] if val_metrics else None,
                'val_acc': val_metrics['acc'] if val_metrics else None,
                'monitor': self.monitor,
                'best_metric': self.best_metric,
                'train_loss': train_results['loss'],
            }
            self.save_model('last.pt', checkpoint=last_checkpoint)

            if should_stop:
                break

        # 训练结束前恢复最佳权重（主流做法：早停只管停训，后续评估用最佳模型）
        # 若不恢复，早停退出时内存中是触发轮次的较差权重，后续 test() 报告会失真
        best_path = self.save_dir / 'best.pt'
        if best_path.exists():
            self.logger.info("📥 Restoring best.pt for subsequent evaluation...")
            self.load_model(str(best_path))
        else:
            self.logger.warning("⚠️ best.pt not found, keeping last-epoch weights")

        # 可视化训练曲线（绘图逻辑见 trainers/visualizer.py，训练数据显式传入）
        if self.train_loss_all and self.val_loss_all:
            self.visualizer.plot_acc_loss(
                train_loss=self.train_loss_all,
                val_loss=self.val_loss_all,
                val_acc=self.val_acc_all,
                val_epochs=self.val_epochs,
                save_path=os.path.join(self.save_dir, 'acc_loss.png'),
            )
        if self.lr_history:
            self.visualizer.plot_lr_history(
                self.lr_history,
                save_path=os.path.join(self.save_dir, 'lr_curve.png'),
            )

        # 标准日志：超参 + 最终指标对照表（供 HPARAMS 面板跨实验对比；
        # 'hparam/' 前缀为业内惯例，与普通标量曲线区分）
        final_metrics = {}
        if self.best_metric not in (float('inf'), float('-inf')):
            final_metrics[f'hparam/best_{self.monitor}'] = self.best_metric
        if self.val_loss_all:
            final_metrics['hparam/final_val_loss'] = self.val_loss_all[-1]
        if self.val_acc_all:
            final_metrics['hparam/final_val_acc'] = self.val_acc_all[-1]
        if final_metrics:
            self.log_hparams(final_metrics)

        # 关闭 TensorBoard writer（由 init_settings 创建，生命周期随训练结束；
        # torch 的 SummaryWriter 关闭后再写入会自动重建 file writer，重复 fit 亦安全）
        if self.writer is not None:
            self.writer.close()

        # 关闭历史记录文件句柄（已逐条 flush，此处仅显式释放资源）
        self.history.close()

        # 测试与训练解耦：由用户在 fit() 结束后手动调用 test()
        self.logger.info("✅ Training finished. Call test() to evaluate on the test set.")


    def train_epoch(self) -> Dict[str, Any]:
        """
        执行一个 epoch 的训练流程：数据搬运、反向传播、优化器/调度器 step、日志记录。
        前向推理与损失计算委托给 training_step（子类可覆写）。
        训练阶段不计算精度指标，只跟踪损失/学习率（指标评估由 evaluate/test 负责）。

        Returns:
            训练结果字典 {'loss', 'time'}
        """
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.model.train()

        if self.current_epoch == 1:
            self.logger.info("start training ...")

        pbar = tqdm(self.train_loader, 
                    desc=f'Epoch {self.current_epoch}/{self.epochs} [Train]', 
                    leave=False,
                    disable=self.pbar_disable,
                    )

        for batch_idx, (inputs, targets) in enumerate(pbar):

            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                targets = targets.squeeze()
                targets =targets.long()

                # 前向（AMP 启用时在 autocast 下以低精度计算）+ 反向传播
                self.optimizer.zero_grad(set_to_none=True) 
                with autocast(device_type=self.device.type,
                              dtype=self.amp_dtype,
                              enabled=self.use_amp):
                    loss = self.training_step(inputs, targets)
                # scaler 禁用时 scale/step/update 退化为普通 backward/step
                self.scaler.scale(loss).backward()

                # 梯度裁剪（防止爆炸，可选；裁剪前需先反缩放梯度）
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm(self.model, self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1

                # ✅ OneCycleLR 需要在 batch 后调用 step()
                if self.scheduler is not None and self.is_batch_scheduler:
                    self.scheduler.step()

                # 损失累积（以设备上的张量累加，避免每个 batch 都 .item() 触发
                # GPU 同步；仅进度条更新时按 log_interval 低频同步）
                batch_size = inputs.size(0)
                total_loss += loss.detach() * batch_size  # 加权累加
                total_samples += batch_size

                # 批次级日志（每 log_interval 个 batch：进度条 + batch 级标准日志，
                # 共用一次 .item() 同步；低频写入也避免 event 文件过大）
                if batch_idx % self.log_interval == 0:
                    batch_loss = loss.item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    self.log_scalars(
                        {'batch_loss': batch_loss, 'batch_lr': current_lr},
                        self.global_step, 'train/')
            except RuntimeError as e:
                # 异常处理：跳过问题 batch，记录日志
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at batch {batch_idx}, skipping...")
                    self._empty_cache()
                    continue
                else:
                    raise e

        # 计算平均损失（全部 batch 因 OOM 被跳过时给出明确报错，而非除零崩溃）
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed in this training epoch "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        avg_loss = total_loss.item() / total_samples  # 加权平均更准确

        # 记录训练元数据（epoch 末重新取 lr，避免 batch 级调度器下的过期值）
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time
        # 标准日志（epoch 级）
        self.log_scalars({
            'epoch_loss': avg_loss,
            'learning_rate': current_lr,
            'samples_per_sec': samples_per_sec,
        }, self.current_epoch, 'train/')

        self.train_loss_all.append(avg_loss)

        # 日志
        self.logger.info(
            f"🏃 Train | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Speed: {samples_per_sec:.0f} samples/sec"
        )

        return  {'loss': avg_loss, 
                 'time': epoch_time}


    def training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        单个 batch 的前向推理 + 损失计算（不含反向传播）。

        子类可覆写此方法实现自定义训练逻辑
        （如多输出模型、多任务损失、深监督等）。

        Args:
            inputs: 输入张量（已在目标设备上）
            targets: 真实标签（已在目标设备上）

        Returns:
            标量损失张量（需保留计算图供 backward）
        """
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return loss

 
    @torch.no_grad()
    def evaluate_epoch(self) -> Dict[str, Any]:
        """
        在验证集上评估模型。
        前向推理、损失计算与指标累积委托给 validation_step（子类可覆写）。

        Returns:
            验证结果字典：{'loss', 'acc', 'time'} + Metrics.compute() 的全部指标
            （如 'oa'、'miou' 等，供统一 monitor 选用）
        """

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(self.val_loader, 
                    desc=f'Epoch {self.current_epoch}/{self.epochs} [Valid]', 
                    leave=False,
                    disable=self.pbar_disable,
                    )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                targets = targets.squeeze()
                targets =targets.long()

                # 前向推理 + 损失 + 指标累积（见 validation_step；AMP 下同样用 autocast 加速）
                with autocast(device_type=self.device.type,
                              dtype=self.amp_dtype,
                              enabled=self.use_amp):
                    loss, _ = self.validation_step(inputs, targets)

                # 加权累加损失（张量累加，避免每 batch 同步）
                batch_size = inputs.size(0)
                total_loss += loss.detach() * batch_size
                total_samples += batch_size

                # 进度条实时更新
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            except RuntimeError as e:
                # 异常处理：跳过问题 batch
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at val batch {batch_idx}, skipping...")
                    self._empty_cache()
                    continue
                else:
                    raise e
        
        # 计算汇总指标（全部 batch 因 OOM 被跳过时给出明确报错）
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed during validation "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        avg_loss = total_loss.item() / total_samples
        results = self.metrics.compute()
        val_acc = results['oa']  # OA: Overall Accuracy

        # 记录元数据
        val_time = time.time() - start_time
        samples_per_sec = total_samples / val_time

        # 标准日志：Metrics.compute() 全量指标（含逐类分组）+ 损失/吞吐
        self.log_val_metrics(results, self.current_epoch)
        self.log_scalars({
            'epoch_loss': avg_loss,
            'epoch_acc': val_acc,
            'samples_per_sec': samples_per_sec,
        }, self.current_epoch, 'val/')
        
        # 更新历史列表（val_epochs 记录对应轮次，eval_interval > 1 时绘图对齐用）
        self.val_loss_all.append(avg_loss)
        self.val_acc_all.append(val_acc)
        self.val_epochs.append(self.current_epoch)
        self.val_metrics_result = results  # 保留详细结果供后续分析
        
        self.logger.info(
            f"🔍 Valid | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Speed: {samples_per_sec:.0f} samples/sec"
        )
        
        # 可选：记录详细指标到 debug 日志
        # self.logger.debug(f"Validation metrics detail: {results}")

        return {**results,
                'loss': avg_loss, 
                'acc': val_acc, 
                'time': val_time}


    def validation_step(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单个 batch 的评估逻辑：前向推理 + 损失计算 + 指标累积。
        验证（evaluate_epoch）与测试（test）共用此钩子，
        子类覆写一处即可同时生效（如多输出模型、自定义指标更新）。

        Args:
            inputs: 输入张量（已在目标设备上）
            targets: 真实标签（已在目标设备上）

        Returns:
            (标量损失张量, logits)；多输出模型覆写时返回主输出 logits，
            供 test() 保存预测结果（argmax / 置信度）使用
        """
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)

        # Metrics 的混淆矩阵在 CPU 上，先搬运避免 GPU 训练时设备不匹配
        # （logits 传入后由 Metrics.update 自动 argmax）
        self.metrics.update(logits.detach().cpu(), targets.detach().cpu())
        return loss, logits


    @torch.no_grad()
    def test(self, 
             report_results: bool = True,
             save_error_index: bool = False,
             save_predictions: bool = False) -> Optional[Dict[str, Any]]:
        """
        在测试集上评估模型（与 fit() 解耦，由用户在训练结束后手动调用）
        
        Args:
            report_results: 是否打印详细测试报告
            save_error_index: 是否保存错误索引
            save_predictions: 是否保存预测结果
        
        Returns:
            测试结果字典 {'loss', 'acc', 'time', 'samples', 'cnf_matrix'}；
            test_loader 未提供时直接返回 None
        """
        if self.test_loader is None:
            self.logger.warning("⚠️ No test dataloader provided, skipping test.")
            return None

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.model.eval()
        self.metrics.reset()

        # 预测结果保存仅支持分类任务（分割输出为逐像素张量，p.item() 会失败）
        if save_predictions and not self.is_classification:
            self.logger.warning("⚠️ save_predictions only supports classification tasks, disabled.")
            save_predictions = False
        # 用于保存预测结果（如果需要）
        predictions = [] if save_predictions else None
        
        # self.logger.info("🎯 Testing model...")
        pbar = tqdm(self.test_loader, 
                    desc='Testing', 
                    leave=True,
                    disable=self.pbar_disable)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                targets = targets.squeeze()
                targets =targets.long()

                # 前向推理 + 损失 + 指标累积（与验证共用 validation_step 钩子，
                # 子类覆写后 test 同步生效）
                with autocast(device_type=self.device.type,
                              dtype=self.amp_dtype,
                              enabled=self.use_amp):
                    loss, outputs = self.validation_step(inputs, targets)
                
                # 加权累加损失（张量累加，避免每 batch 同步）
                batch_size = inputs.size(0)
                total_loss += loss.detach() * batch_size
                total_samples += batch_size

                # 保存预测结果（用于后续分析/提交）
                if save_predictions and predictions is not None:
                    # 记录: (global_index, prediction, target, confidence)
                    preds = torch.argmax(outputs, dim=1).detach().cpu()
                    targets_cpu = targets.detach().cpu()
                    confidences = torch.softmax(outputs, dim=1).max(dim=1).values.detach().cpu()
                    # 全局索引用累计样本数推算：不依赖 loader.batch_size
                    # （batch_sampler 下为 None），OOM 跳过的 batch 不占用索引
                    base_idx = total_samples - batch_size
                    for i, (p, t, c) in enumerate(zip(preds, targets_cpu, confidences)):
                        predictions.append({
                            'index': base_idx + i,
                            'pred': p.item(),
                            'target': t.item(),
                            'confidence': c.item(),
                            'correct': p.item() == t.item()
                        })
                
                # 进度条实时更新
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
            except RuntimeError as e:
                # 异常处理：跳过问题 batch
                if "out of memory" in str(e):
                    self.logger.warning(f"⚠️ OOM at test batch {batch_idx}, skipping...")
                    self._empty_cache()
                    continue
                else:
                    raise e
        
        # 计算汇总指标（全部 batch 因 OOM 被跳过时给出明确报错）
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed during testing "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        avg_loss = total_loss.item() / total_samples
        results = self.metrics.compute()
        test_acc = results.get('oa', 0.0)
        # 混淆矩阵从指标计算器直接获取，转 numpy 供绘图使用
        cnf_matrix = self.metrics.confusion_matrix.cpu().numpy()
        self.cnf_matrix = cnf_matrix

        test_time = time.time() - start_time
        samples_per_sec = total_samples / test_time

        # 保存预测结果（如果需要）
        if save_predictions and predictions:
            import pandas as pd
            save_path = os.path.join(self.save_dir, 'test_predictions.csv')
            df = pd.DataFrame(predictions)
            df.to_csv(save_path, index=False)
            self.logger.info(f"📁 Predictions saved to {save_path}")
            
            # 可选：保存错误样本索引（用于错误分析）
            if save_error_index:
                errors = [p['index'] for p in predictions if not p['correct']]
                if errors:
                    error_path = os.path.join(self.save_dir, 'test_errors.txt')
                    with open(error_path, 'w') as f:
                        f.write('\n'.join(map(str, errors)))
                    self.logger.info(f"❌ {len(errors)} errors logged to {error_path}")
        
        # 打印详细测试报告（报告格式化见 trainers/visualizer.py）
        if report_results:
            self.visualizer.print_test_report(
                results, cnf_matrix, test_time, samples_per_sec,
                is_classification=self.is_classification,
            )

        # 混淆矩阵可视化（随测试一起从 fit() 移入，仅分类任务绘制）
        if self.is_classification:
            self.visualizer.plot_confusion_matrix(
                cm=cnf_matrix,
                normalize=False,
                save_path=self.save_dir / 'confusion_matrix.png'
            )
            self.visualizer.plot_confusion_matrix(
                cm=cnf_matrix,
                normalize=True,
                save_path=self.save_dir / 'confusion_matrix_normalized.png'
            )
        
        return {'loss': avg_loss, 
                'acc': test_acc, 
                'time': test_time, 
                'samples': total_samples, 
                'cnf_matrix': cnf_matrix,
                }


    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        对输入数据进行预测
        
        Args:
            inputs: 输入张量（支持单样本 (C, H, W) 或批量 (N, C, H, W)）
        
        Returns:
            预测类别标签
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        if inputs.dim() == 3:  # (C, H, W)
            inputs = inputs.unsqueeze(0)  # (1, C, H, W)
        inputs = inputs.to(self.device)
        self.model.eval()
        # 与验证/测试一致走 autocast（AMP 启用时），推理路径与评估对齐
        with autocast(device_type=self.device.type,
                      dtype=self.amp_dtype,
                      enabled=self.use_amp):
            logits = self.model(inputs)
        return torch.argmax(logits, dim=1)


    def _unwrap_model(self) -> nn.Module:
        """
        获取原始模型。

        torch.compile 会将模型包装为 OptimizedModule，其 state_dict 的 key
        带 ``_orig_mod.`` 前缀；保存/加载时统一使用内层原始模型，
        保证 checkpoint 在 compile / 非 compile 模式下可互相加载。
        """
        return getattr(self.model, '_orig_mod', self.model)


    def save_model(self, filename: str, checkpoint: Optional[Dict[str, Any]] = None) -> str:
        """
        保存模型检查点
        
        Args:
            filename: 文件名 ('last.pt' 或 'best.pt')
            checkpoint: 检查点字典（包含 model、optimizer、scheduler 等）
        
        Returns:
            保存的文件路径
        """
        model_path = self.save_dir / filename
        
        # 原子保存：先写临时文件再 os.replace 覆盖，
        # 避免保存中途进程被 kill 时新旧 checkpoint 一起损坏
        tmp_path = model_path.with_suffix(model_path.suffix + '.tmp')
        try:
            if checkpoint:
                torch.save(checkpoint, tmp_path)
            else:
                # 仅保存模型参数（用于轻量级部署，unwrap 避免 _orig_mod. 前缀）
                torch.save(self._unwrap_model().state_dict(), tmp_path)
            os.replace(tmp_path, model_path)
        except BaseException:
            # 保存失败/中断时清理残留临时文件，原 checkpoint 不受影响
            tmp_path.unlink(missing_ok=True)
            raise
        
        self.logger.info(f"💾 Model saved: {filename}")
        return str(model_path)


    def _restore_history(self, src_log: Path) -> None:
        """
        断点续训时衔接训练历史（曲线数据由 History 组件以 JSONL 持久化）。

        从旧运行目录的 training_log.jsonl 读取记录，按 checkpoint 的 epoch 截断
        （丢弃比权重快照更晚的“幽灵”记录，如崩溃前多写的日志行、
        或从较早的 best.pt 恢复时其之后的记录），迁移到本次运行的新日志文件，
        并重建内存中的绘图列表，使续训后绘制的曲线完整衔接。

        Args:
            src_log: checkpoint 同目录的 training_log.jsonl 路径，
                     不存在时保持空历史（仅绘制续训段）
        """
        if not src_log.exists():
            self.logger.warning(
                "⚠️ No training_log.jsonl found next to checkpoint, "
                "history curves start fresh"
            )
            return
        if src_log.resolve() == self.history.log_path.resolve():
            return  # 同一文件无需迁移（输出目录按时间戳隔离，正常不会发生）

        old = History(src_log)
        old.load()
        records = [r for r in old.records if r.get('epoch', 0) <= self.start_epoch]
        for record in records:
            self.history.append(record)

        # 重建内存绘图列表（与 fit 循环中的追加逻辑对齐）
        def pick(phase: str, key: str) -> list:
            return [r[key] for r in records if r.get('phase') == phase and key in r]

        self.train_loss_all = pick('train', 'loss')
        self.lr_history = pick('train', 'lr')
        self.val_loss_all = pick('val', 'loss')
        self.val_acc_all = pick('val', 'acc')
        self.val_epochs = pick('val', 'epoch')
        self.logger.info(
            f"📈 Training history restored from {src_log}: "
            f"{len(self.train_loss_all)} train epochs, "
            f"{len(self.val_epochs)} validation rounds"
        )


    def load_model(self, checkpoint_fn: Optional[str] = None, resume: bool = False) -> str:
        """
        加载模型检查点
        
        Args:
            checkpoint_fn: 检查点文件路径
                        如果为 None，则自动按优先级查找: last.pt → best.pt
            resume: 是否为断点续训。True 时额外恢复优化器/调度器状态、
                    start_epoch、best_metric 和训练历史曲线；False 时仅加载模型权重
                    （如训练结束后恢复 best.pt 做最终评估）。
        """
        # 自动查找检查点
        if checkpoint_fn is None:
            last_path = self.save_dir / 'last.pt'
            best_path = self.save_dir / 'best.pt'
            
            if last_path.exists():
                checkpoint_fn = str(last_path)
                self.logger.info("🔍 Auto-loading last.pt for resume training")
            elif best_path.exists():
                checkpoint_fn = str(best_path)
                self.logger.info("🔍 Auto-loading best.pt for inference")
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.save_dir}")
        
        if not os.path.exists(checkpoint_fn):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_fn} not found.")

        try:
            checkpoint = torch.load(checkpoint_fn, weights_only=True, map_location=self.device)

            # 兼容两种格式：完整 checkpoint 字典 / 纯 state_dict
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            # 去除 torch.compile 产生的 _orig_mod. 前缀（兼容旧 checkpoint）
            state_dict = {
                (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                for k, v in state_dict.items()
            }

            # 加载到内层原始模型，避免 compile 包装导致的 key 不匹配；
            # strict=False 但显式检查匹配情况，杜绝静默加载失败
            target_model = self._unwrap_model()
            incompatible = target_model.load_state_dict(state_dict, strict=False)
            missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
            model_keys = set(target_model.state_dict().keys())
            loaded_count = len(model_keys) - len(missing)
            if loaded_count == 0:
                raise RuntimeError(
                    f"No weights loaded from {checkpoint_fn}: all {len(model_keys)} "
                    f"model keys missing (checkpoint has {len(state_dict)} keys). "
                    f"Checkpoint and model architecture do not match."
                )
            if missing or unexpected:
                self.logger.warning(
                    f"⚠️ Partial state_dict load: {loaded_count}/{len(model_keys)} keys loaded | "
                    f"missing: {len(missing)} | unexpected: {len(unexpected)}"
                )
                if missing:
                    self.logger.warning(f"   • Missing keys (first 5): {missing[:5]}")
                if unexpected:
                    self.logger.warning(f"   • Unexpected keys (first 5): {unexpected[:5]}")

            # 断点续训：恢复训练状态（仅加载权重时不动优化器/进度，
            # 避免最终评估恢复 best.pt 时污染训练状态）
            if resume and isinstance(checkpoint, dict):
                if 'optimizer' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

                if 'lr_schedule' in checkpoint and checkpoint['lr_schedule'] and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['lr_schedule'])

                # 恢复 AMP 损失缩放器状态（仅 CUDA + AMP 训练的 checkpoint 才有）
                if checkpoint.get('scaler') and self.scaler.is_enabled():
                    self.scaler.load_state_dict(checkpoint['scaler'])

                # checkpoint 中 epoch 为已完成轮次，fit 从该索引继续（current_epoch = epoch + 1）
                self.start_epoch = checkpoint.get('epoch', 0) or 0
                # 恢复历史最佳监控值：优先取新格式 best_metric（last.pt 也会记录）；
                # 旧 checkpoint 仅有 best_val_acc / val_acc（acc 语义，仅 monitor='acc' 时可复用）
                ckpt_monitor = checkpoint.get('monitor')
                restored_best = checkpoint.get('best_metric')
                if restored_best is None and self.monitor == 'acc':
                    restored_best = checkpoint.get('best_val_acc')
                    if restored_best is None:
                        restored_best = checkpoint.get('val_acc')
                if ckpt_monitor is not None and ckpt_monitor != self.monitor:
                    self.logger.warning(
                        f"⚠️ Checkpoint monitor '{ckpt_monitor}' != current "
                        f"'{self.monitor}', best_metric not restored (starting fresh)"
                    )
                elif restored_best is not None:
                    # 按 monitor_mode 方向取更优值，避免 best.pt 被更差模型覆盖
                    better = min if self.monitor_mode == 'min' else max
                    self.best_metric = better(self.best_metric, restored_best)

                # 恢复训练历史曲线：从 checkpoint 同目录的 training_log.jsonl 迁移
                # （曲线数据由 History 组件持久化，不入 checkpoint）
                self._restore_history(Path(checkpoint_fn).parent / 'training_log.jsonl')

                self.logger.info(
                    f"🔄 Training state restored | "
                    f"start_epoch: {self.start_epoch} | "
                    f"best {self.monitor}: {self.best_metric:.4f}"
                )

            # ✅ 加载时打印关键指标（替代文件名中的信息）
            epoch = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
            val_acc = checkpoint.get('val_acc') if isinstance(checkpoint, dict) else None
            val_loss = checkpoint.get('val_loss') if isinstance(checkpoint, dict) else None
            
            self.logger.info(f"📥 Model loaded from {os.path.basename(checkpoint_fn)}")
            self.logger.info(f"   • Epoch: {epoch}")
            if val_acc is not None:
                self.logger.info(f"   • Val Acc: {val_acc:.4f}")
            if val_loss is not None:
                self.logger.info(f"   • Val Loss: {val_loss:.4f}")
            
            return checkpoint_fn
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise e


    def resume_training(self, checkpoint_fn: Optional[str] = None) -> None:
        """
        从检查点恢复训练
        
        Args:
            checkpoint_fn: 检查点路径，None 时自动查找 last.pt
        """
        loaded_path = self.load_model(checkpoint_fn, resume=True)
        self.logger.info(f"🔄 Resuming training from {loaded_path}")
        
        # 继续执行训练（fit 会从 self.start_epoch 开始）
        self.fit()


    def export_onnx(self,
                    output_path: Optional[str] = None,
                    input_shape: Optional[Tuple[int, ...]] = None,
                    opset_version: int = 17) -> str:
        """
        导出 ONNX 模型（batch 维动态，单样本形状固定）

        Args:
            output_path: 输出文件路径，None 时保存到 save_dir/model.onnx
            input_shape: 单样本形状（不含 batch 维，如 (3, 224, 224)）；
                         None 时从 train/val/test 任一可用 loader 取样推断
            opset_version: ONNX 操作集版本

        Returns:
            导出的文件路径
        """
        # 推断输入形状：优先显式指定，否则从任一可用 loader 取样
        # （纯推理场景可能没有 train_loader）
        if input_shape is None:
            loader = self.train_loader or self.val_loader or self.test_loader
            if loader is None:
                raise ValueError(
                    "input_shape not provided and no dataloader available to infer it"
                )
            sample, _ = next(iter(loader))
            input_shape = tuple(sample.shape[1:])

        if output_path is None:
            output_path = str(self.save_dir / 'model.onnx')

        # 导出原始模型（torch.compile 的包装层不可序列化）；
        # eval 模式固化 BN/Dropout 行为
        model = self._unwrap_model()
        model.eval()
        dummy_input = torch.randn(1, *input_shape, device=self.device)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=opset_version,
        )
        self.logger.info(f"✅ Model exported to ONNX: {output_path}")
        return output_path

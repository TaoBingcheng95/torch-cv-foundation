
import logging
import os
import shutil
import gc
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
# 混合精度训练
from torch.amp import autocast
from torch.amp import GradScaler

from .visualizer import Visualizer
from .report import print_test_report
from .utils import EarlyStopping, History
from .tb_logger import TensorBoardLogger

from optimizers import clip_grad_norm
from utils.hardware import select_device, collect_hardware_report
from utils.logger import get_logger, add_file_handler

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
                 device: Union[str, torch.device] = 'auto',  # 'auto' | 'cuda' | 'cpu' | 'mps'，也可直接传 torch.device
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 class_names: Optional[List[str]] = None,
                 is_classification: bool = True, # 是否为分类任务（影响指标计算和日志记录）
                 num_classes: int = 10,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[lr_scheduler._LRScheduler] = None,
                 criterion: Optional[nn.Module] = None,  # None 时默认 CrossEntropyLoss
                 metric = None,
                 max_grad_norm: Optional[float] = None,  # 梯度裁剪
                 max_epochs: int = 30,
                 min_epochs: int = 5,  # 最小训练轮数（早停保护期内不触发）
                 log_interval: int = 5,
                 eval_interval: int = 1,  # 每隔多少个 epoch 验证一次（1 = 每轮都验证）
                 resume: Optional[str]=None,
                 compile_model:bool = False, 
                 use_amp: bool = False,  # 混合精度训练（AMP）
                 early_stop_patience: Optional[int] = 5,  # 早停容忍次数（None 表示禁用早停）
                 early_stop_delta: float = 0.0,  # 早停判定的最小改善阈值
                 monitor: str = 'val/loss',  # 统一监控指标：best.pt / 早停 / Plateau 共用
                 monitor_mode: str = 'auto',  # 'auto' | 'min' | 'max'
                 output_dir: str='./output',
                #  logger:logging.Logger = None,
                 use_tensorboard: bool = True,):  # **kwargs     
        """
        初始化训练器
        
        :param optimizer: 已实例化的优化器（由 build_optimizer 或外部构建）
        :param scheduler: 已实例化的调度器（由 build_scheduler 或外部构建，None 表示固定学习率）
        :param early_stop_patience: 早停容忍次数。语义为"连续 N 次验证不改善后停训"
            （与 PyTorch Lightning 一致），不是训练 epoch 数。
            实际等效 epoch 数 = early_stop_patience × eval_interval。
            例如 eval_interval=5、patience=5 时，至少连续 25 个 epoch 不改善才会停。
            None 表示禁用早停。搭配 ReduceLROnPlateau 时应大于其 patience
            （建议 2~3 倍），否则 LR 还没来得及衰减就会触发早停。
        :param monitor: 统一监控指标，best.pt 保存 / 早停 / ReduceLROnPlateau 共用。
            必须使用 slash 前缀风格，与 metrics.csv 列名一致：
            'val/loss'、'val/acc' 或 metrics.compute() 结果中的任意键加 'val/' 前缀
            （分类如 'val/f1'，分割如 'val/iou'）
        :param monitor_mode: 'auto' 按名称推断
            （含 loss/err/perplexity → min；含 acc/f1/iou/precision/recall/kappa/oa → max），
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
        # 显式指定不可用后端时 fail fast；
        self.device = select_device(device)

        # 输出目录
        self.save_dir = Path(os.path.join(output_dir, self.timestamp))

        # 核心组件
        # _wrap_model 钩子：默认原样返回，DDPTrainer 在此套 DistributedDataParallel；
        # 须在 .to(device) 之前包装，使 DDP 的梯度钩子挂在已绑定设备的模型上
        self.model = self._wrap_model(model).to(self.device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

        self.num_classes = num_classes
        self.class_names = class_names or [f'Class-{i}' for i in range(num_classes)]
        self.is_classification = is_classification

        # 优化器与调度器（由调用方外部实例化后传入，训练器只负责使用与状态管理）
        self.optimizer = optimizer
        self.scheduler = scheduler
        # 是否为 batch 级调度器（如 OneCycleLR），按实例类型判定
        self.is_batch_scheduler = isinstance(self.scheduler, lr_scheduler.OneCycleLR)
        # 损失函数（默认在此实例化，避免可变默认参数被多个实例共享；
        # 显式 to(device)：带 weight 等 buffer 的 loss 需与模型同设备）
        self.criterion = (criterion or nn.CrossEntropyLoss()).to(self.device)
        # 指标模板（由调用方传入基于 torchmetrics 的 MetricCollection 实例，
        # 内部通过 clone(prefix) 为 val/test 创建独立的指标计算器）
        self._metric_template = metric
        self.max_grad_norm = max_grad_norm

        # 任务配置
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.log_interval = log_interval
        self.eval_interval = max(1, eval_interval)
        # 当前 epoch
        self.current_epoch = 0
        # 起始 epoch（断点续训时由 load_checkpoint 恢复）
        self.start_epoch = 0

        # 统一监控指标：best.pt 保存、早停、ReduceLROnPlateau 共用同一 monitor
        # 强制 slash 前缀风格（'val/loss'、'val/acc'），与 metrics.csv 列名一致；
        # 内部访问 val_metrics 时用 _resolve_monitor_key 取原始键名（'loss'、'acc'）
        if '/' not in monitor:
            raise ValueError(
                f"monitor must use slash-prefix style (e.g. 'val/loss', 'val/acc'), "
                f"got '{monitor}'"
            )
        self.monitor = monitor
        # mode 推断：按 monitor 末段关键字匹配（不区分大小写）
        if monitor_mode == 'auto':
            key = monitor.split('/')[-1].lower()
            MIN_KEYS = ('loss', 'err', 'perplexity')
            MAX_KEYS = ('acc', 'f1', 'iou', 'precision', 'recall', 'kappa', 'oa')
            if any(k in key for k in MIN_KEYS):
                self.monitor_mode = 'min'
            elif any(k in key for k in MAX_KEYS):
                self.monitor_mode = 'max'
            else:
                raise ValueError(
                    f"Cannot infer monitor_mode for monitor='{monitor}' "
                    f"(key='{key}' not in MIN_KEYS={MIN_KEYS} or MAX_KEYS={MAX_KEYS}), "
                    f"please specify monitor_mode='min'/'max' explicitly"
                )
        elif monitor_mode in ('min', 'max'):
            self.monitor_mode = monitor_mode
        else:
            raise ValueError(
                f"monitor_mode must be 'auto'/'min'/'max', got '{monitor_mode}'")
        # 历史最佳监控值（断点续训时由 load_checkpoint 恢复，避免 best.pt 被更差模型覆盖）
        self.best_metric = float('inf') if self.monitor_mode == 'min' else float('-inf')

        # 全局步数（batch 级 TensorBoard 曲线的横轴，断点续训时在 fit 中推算衔接）
        self.global_step = 0

        # 早停配置（与调度器配置解耦，patience=None 表示禁用早停）
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        # 早停器状态（断点续训时由 load_checkpoint 恢复，fit 中创建 early_stopper 后消费）
        self._early_stopper_state: Optional[Dict[str, Any]] = None

        # 恢复训练
        self.resume = resume
        # 编译选项（仅在 CUDA 上有效）
        self.compile_model = compile_model

        # 混合精度训练（AMP）：autocast 精度按设备选择，
        # GradScaler 仅 CUDA + fp16 需要；其余设备构造 disabled 的 scaler，
        # 其 scale/step/update 自动退化为透传，训练循环无需分支
        self.use_amp = use_amp
        # MPS 防护：MPS 的 fp16 autocast 可用但没有 GradScaler，小梯度会静默
        # 下溢为 0——损失照常打印、训练看似在跑，实际权重几乎不更新，验证时
        # sigmoid 峰值过不了 com_threshold，F1 等指标全为 0。与其靠用户事后
        # 看告警排查，这里直接强制关闭（记录标志，待 logger 就绪后说明）
        self._amp_forced_off = False
        if self.use_amp and self.device.type == 'mps':
            self.use_amp = False
            self._amp_forced_off = True
        self.amp_dtype = torch.bfloat16 if self.device.type == 'cpu' else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == 'cuda')

        # 日志器
        self.logger = logger
        # TensorBoard 日志组件（writer 在 init_settings 中创建，
        # 因 save_dir 需先建目录；DDP 非主进程不走 init_settings，天然不创建）
        self.use_tensorboard = use_tensorboard
        self.tb_logger: Optional[TensorBoardLogger] = None
        # 进度条开关（DDP 非主进程置 True 避免多进程进度条交错刷屏）
        self.pbar_disable = False

        # 可视化组件（训练过程中动态绘制训练曲线）
        # 在 init_settings 中创建
        self.visualizer = None

        # 指标记录（内存列表供绘图；持久化由 History 组件写入 CSV（metrics.csv），
        # 断点续训时由 load_checkpoint 从旧目录的日志恢复，保证曲线完整衔接；
        # 全新训练应新建 Trainer 实例）
        # val_metrics_history: 验证指标历史，键名与 metrics.csv 列名一致
        # （'val/loss'、'val/acc'、'val/f1' 等）。绘图时按需取用
        self.val_metrics_history: Dict[str, list] = {}
        self.train_loss_all = []
        self.train_epochs = []
        self.val_loss_all = []
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

        # 清理上一个 trainer 残留的 FileHandler，避免日志串文件
        # （self.logger 是模块级单例，多实例共享；add_file_handler 按路径去重，
        # 但不同 save_dir 的 train.log 路径不同，无法被去重，会累积）
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)
                h.close()

        add_file_handler(self.logger, self.save_dir / "train.log")

        # 训练历史记录器（CSV 追加 + flush，崩溃安全；Lightning 风格表头）
        self.history = History(
            self.save_dir / 'metrics.csv',
            fieldnames=self._build_history_fieldnames(),
        )

        # TensorBoard 日志组件（与 checkpoint 同目录便于归档对比；
        # 查看：tensorboard --logdir <output_dir>，各时间戳子目录自动识别为 run）
        tb_dir = self.save_dir / 'tensorboard'
        self.tb_logger = TensorBoardLogger(
            log_dir=str(tb_dir),
            enabled=self.use_tensorboard,
        )

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

        # 优化器和调度器：类型检测 + 配置摘要日志
        self.logger.info(f"📌 Monitor: {self.monitor} (mode: {self.monitor_mode})")
        self._detect_scheduler_type()
        self._log_component_configs()

        # 恢复训练（resume=True: 同时恢复 epoch/best_metric/优化器/调度器状态）
        if self.resume:
            self.logger.info(f"📥 Resuming from checkpoint: {self.resume}")
            self.load_checkpoint(self.resume, resume=True)

        # 指标计算器（基于 torchmetrics MetricCollection 的 clone 机制，
        # 从用户传入的模板派生出 val/test 两套独立实例，各自带 phase 前缀；
        # state 跟随训练设备，DDP 下须在通信设备上）
        self.logger.info("📊 Initializing metrics calculator...")
        if self._metric_template is None:
            raise ValueError(
                "metric 参数必须传入基于 torchmetrics 的指标模板"
                "（如 MulticlassClassificationMetric / MulticlassSegmentationMetric）"
            )
        # 模板应为 phase 中性（无 prefix/postfix），trainer 负责按 val/test 分配前缀；
        # 若模板已携带 prefix，clone(prefix='val/') 会覆盖，但 fieldnames 可能与实际键名不一致
        if getattr(self._metric_template, 'prefix', None):
            raise ValueError(
                "metric 模板不应携带 prefix（ trainer 自动分配 'val/'/'test/'），"
                f"当前 prefix='{self._metric_template.prefix}'，请移除后重试"
            )
        if getattr(self._metric_template, 'postfix', None):
            self.logger.warning(
                f"⚠️ metric 模板携带 postfix='{self._metric_template.postfix}'，"
                "可能影响键名匹配，建议移除"
            )
        self._val_metric = self._metric_template.clone(prefix='val/')
        self._test_metric = self._metric_template.clone(prefix='test/')
        self._val_metric.to(self.device)
        self._test_metric.to(self.device)
        self.metrics = self._val_metric  # 默认活跃指标为 val

        # 可视化器：只持有展示配置（输出目录/类别名），训练数据由调用时显式传入
        self.visualizer = Visualizer(
            save_dir=self.save_dir,
            class_names=self.class_names,
            logger=self.logger,
        )

        # 模型编译（PyTorch 2.0+）
        if self.compile_model:
            try:
                self.logger.info("Compiling model with torch.compile...")
                self.model = torch.compile(self.model)
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}, using original model")
        self.logger.info("✅ Initialization complete!")


    def _detect_scheduler_type(self) -> None:
        """
        检测调度器类型并执行一致性校验。

        - 设置 is_batch_scheduler（OneCycleLR 按 batch step，其余按 epoch step）
        - ReduceLROnPlateau 的 mode 与 monitor_mode 对齐检查
        """
        if self.scheduler is None:
            self.logger.info("Scheduler: None (using constant learning rate)")
            return

        self.logger.info(f"Scheduler: {type(self.scheduler).__name__}")

        # Plateau 与统一 monitor 的 mode 对齐检查（不一致时 LR 衰减方向会反）
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau) and \
                self.scheduler.mode != self.monitor_mode:
            self.logger.warning(
                f"⚠️ ReduceLROnPlateau mode '{self.scheduler.mode}' != monitor_mode "
                f"'{self.monitor_mode}' (monitor='{self.monitor}'), "
                f"please align scheduler mode with monitor_mode='{self.monitor_mode}'"
            )


    def _log_component_configs(self) -> None:
        """
        记录各组件的配置摘要并保存为 JSON（可扩展）。

        行为：
          1. 从实例提取结构化配置字典（与 optim_cfg/sched_cfg 同构，可直接用于复现）
          2. 输出可读日志到控制台
          3. 写入 save_dir/component_configs.json

        当前支持：
          - Optimizer：类型、参数组、学习率、权重衰减等
          - Scheduler：类型及关键参数

        后续可扩展至 criterion、metric 等组件的配置记录。
        """
        config = self._extract_component_configs()

        # ── 控制台日志 ─────────────────────────────────────────────────────
        self.logger.info("=" * 50)
        self.logger.info("📋 Component Configuration Summary")
        self.logger.info("=" * 50)

        # Optimizer
        opt = config.get('optimizer')
        if opt:
            self.logger.info(f"[Optimizer] {opt['type']}")
            for i, pg in enumerate(opt.get('param_groups_info', [])):
                group_info = f"  Group {i}: lr={pg['lr']:.2e}, weight_decay={pg['weight_decay']:.2e}"
                if 'betas' in pg:
                    group_info += f", betas={pg['betas']}"
                if 'momentum' in pg and pg['momentum'] != 0:
                    group_info += f", momentum={pg['momentum']}"
                self.logger.info(group_info)
        else:
            self.logger.info("[Optimizer] None")

        # Scheduler
        sched = config.get('scheduler')
        if sched:
            sched_info = f"[Scheduler] {sched['type']}"
            # 拼接非 type 字段为可读摘要
            detail = {k: v for k, v in sched.items() if k != 'type'}
            if detail:
                sched_info += " | " + ", ".join(f"{k}={v}" for k, v in detail.items())
            self.logger.info(sched_info)
        else:
            self.logger.info("[Scheduler] None (constant learning rate)")

        self.logger.info("=" * 50)

        # ── 保存 JSON ───────────────────────────────────────────────────────
        config_path = self.save_dir / 'component_configs.json'
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 Component configs saved: {config_path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save component configs: {e}")


    def _extract_component_configs(self) -> Dict[str, Any]:
        """
        从 optimizer/scheduler 实例提取结构化配置字典。

        返回格式与 build_optimizer/build_scheduler 的 cfg 入参同构，
        可直接作为复现时的配置输入。

        Returns:
            包含 'optimizer' 和 'scheduler' 键的字典
        """
        config: Dict[str, Any] = {}

        # ── Optimizer ──────────────────────────────────────────────────────
        if self.optimizer is not None:
            opt_cfg: Dict[str, Any] = {
                'type': type(self.optimizer).__name__.lower(),
            }
            # 从第一个参数组提取通用参数
            pg0 = self.optimizer.param_groups[0]
            opt_cfg['lr'] = pg0['lr']
            opt_cfg['weight_decay'] = pg0['weight_decay']
            if 'betas' in pg0:
                opt_cfg['betas'] = list(pg0['betas'])
            if 'momentum' in pg0:
                opt_cfg['momentum'] = pg0['momentum']
            if 'alpha' in pg0:
                opt_cfg['alpha'] = pg0['alpha']
            if 'eps' in pg0:
                opt_cfg['eps'] = pg0['eps']
            if len(self.optimizer.param_groups) > 1:
                opt_cfg['head_lr_scale'] = (
                    self.optimizer.param_groups[1]['lr'] / pg0['lr']
                    if pg0['lr'] > 0 else 1.0
                )
            # 保存每个参数组的详细信息（仅供日志展示，不用于复现）
            opt_cfg['param_groups_info'] = [
                {k: v for k, v in pg.items()
                 if k in ('lr', 'weight_decay', 'betas', 'momentum')}
                for pg in self.optimizer.param_groups
            ]
            config['optimizer'] = opt_cfg

        # ── Scheduler ───────────────────────────────────────────────────────
        if self.scheduler is not None:
            sched_cfg: Dict[str, Any] = {}
            # 调度器类 → builder 工厂键名的反向映射
            # 用于将实例类型还原为 build_scheduler 可识别的 type 字段
            _SCHED_CLASS_TO_KEY = {
                lr_scheduler.StepLR: 'steplr',
                lr_scheduler.MultiStepLR: 'multisteplr',
                lr_scheduler.ExponentialLR: 'exponentiallr',
                lr_scheduler.ReduceLROnPlateau: 'plateau',
                lr_scheduler.CosineAnnealingLR: 'cosineannealinglr',
                lr_scheduler.OneCycleLR: 'onecyclelr',
                lr_scheduler.LambdaLR: 'warmup_cosine',  # 项目中 LambdaLR 实例均来自 warmup_cosine
            }
            sched_cls = type(self.scheduler)
            sched_cfg['type'] = _SCHED_CLASS_TO_KEY.get(sched_cls, sched_cls.__name__.lower())

            # 按类型提取关键参数（与 build_scheduler 的 sched_kwargs 对齐）
            if isinstance(self.scheduler, lr_scheduler.StepLR):
                sched_cfg['step_size'] = self.scheduler.step_size
                sched_cfg['gamma'] = self.scheduler.gamma
            elif isinstance(self.scheduler, lr_scheduler.MultiStepLR):
                sched_cfg['milestones'] = list(self.scheduler.milestones)
                sched_cfg['gamma'] = self.scheduler.gamma
            elif isinstance(self.scheduler, lr_scheduler.ExponentialLR):
                sched_cfg['gamma'] = self.scheduler.gamma
            elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                sched_cfg['mode'] = self.scheduler.mode
                sched_cfg['factor'] = self.scheduler.factor
                sched_cfg['patience'] = self.scheduler.patience
            elif isinstance(self.scheduler, lr_scheduler.CosineAnnealingLR):
                sched_cfg['T_max'] = self.scheduler.T_max
                sched_cfg['eta_min'] = self.scheduler.eta_min
            elif isinstance(self.scheduler, lr_scheduler.OneCycleLR):
                sched_cfg['max_lr'] = self.scheduler.max_lrs[0]
                sched_cfg['total_steps'] = self.scheduler.total_steps
            config['scheduler'] = sched_cfg

        return config


    def _build_history_fieldnames(self) -> List[str]:
        """
        构造 CSV 表头列名（Lightning 风格 slash 前缀）。

        固定元数据列 + train 专属列 + val 通用列 + 从指标模板动态获取的汇总列。
        逐类指标不入 CSV（走 TensorBoard 的 val_per_class/ 分组），
        top_k 暂不写入（后续按需扩展）。

        列结构（按出现顺序）：
            epoch, phase
            train/loss, train/lr, train/time
            val/loss, val/acc, val/time
            val/<summary_metrics...>   # 由 metric_template.metric_keys() 动态生成
        """
        # 1. 元数据（train/val 行均填）
        fields = ['epoch', 'phase']
        # 2. train 专属（train 行填，val 行留空）
        fields += ['train/loss', 'train/lr', 'train/time']
        # 3. val 通用（val 行填，train 行留空）；val/acc 统一对应分类 acc / 分割 oa
        fields += ['val/loss', 'val/acc', 'val/time']
        # 4. 从指标模板动态获取汇总指标键名（已含 val/ 前缀），
        #    替代硬编码以适配不同指标模板（分类/分割/自定义）
        if self._metric_template is not None:
            val_keys = self._metric_template.clone(prefix='val/').metric_keys()
            fields += [k for k in val_keys if k not in fields]
        return fields


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


    def _resolve_monitor_key(self, val_metrics: Dict[str, Any]) -> Tuple[str, float]:
        """
        从 val_metrics 中取出 monitor 对应的键名与指标值。
    
        validation_epoch 返回的键统一采用 slash 前缀风格（'val/loss'、'val/acc' 等），
        与 self.monitor 直接匹配，无需短名兼容。
    
        Args:
            val_metrics: validation_epoch 返回的指标字典（键已含 'val/' 前缀）
    
        Returns:
            (键名, 指标值)
    
        Raises:
            KeyError: monitor 在 val_metrics 中找不到
        """
        if self.monitor in val_metrics:
            return self.monitor, val_metrics[self.monitor]
        raise KeyError(
            f"monitor '{self.monitor}' not found in val_metrics, "
            f"available keys: {sorted(val_metrics.keys())}"
        )


    def _is_better(self, value: float) -> bool:
        """判断监控指标是否优于历史最佳（按 monitor_mode 方向比较）

        NaN 防护：NaN 与任何值比较均为 False，避免 best.pt 被异常值（如除零
        产生的 NaN）污染——NaN 永远不视为更优，也不会触发保存。
        """
        if value != value:  # NaN: x != x 为 True
            return False
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
            # mode 对齐已在 _detect_scheduler_type 中校验
            # monitor 采用 slash 前缀风格，这里归一化取原始键
            _, metric = self._resolve_monitor_key(val_metrics)

            self.scheduler.step(metric)
            self.logger.debug(
                f"ReduceLROnPlateau step: {self.monitor}={metric:.4f}"
            )
        else:
            self.scheduler.step()
        return self.optimizer.param_groups[0]['lr']


    def _append_val_metrics(self, results: Dict[str, Any], val_time: float) -> None:
        """
        将本轮验证的指标追加到 val_metrics_history。

        validation_epoch 返回的键统一采用 val/ 前缀风格，
        直接作为 CSV 列名存入内存 dict；非标量值（混淆矩阵等）自动过滤。

        Args:
            results: validation_epoch 返回的指标字典（键已含 'val/' 前缀）
            val_time: 验证耗时（秒）
        """
        def _ensure(key):
            if key not in self.val_metrics_history:
                self.val_metrics_history[key] = []

        # val/time 不在 compute() 输出中，单独追加
        _ensure('val/time'); self.val_metrics_history['val/time'].append(val_time)

        # 其余指标键统一为 val/ 前缀，直接写入；过滤非标量值
        for key, value in results.items():
            if not isinstance(value, (int, float)):
                if hasattr(value, 'ndim') and value.ndim == 0:
                    value = value.item()
                else:
                    continue
            _ensure(key)
            self.val_metrics_history[key].append(value)


    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """模型包装钩子：默认原样返回，DDPTrainer 在此套 DistributedDataParallel"""
        return model

    def _unwrap_model(self) -> nn.Module:
        """
        获取原始模型。

        包装顺序为 ``torch.compile(DistributedDataParallel(model))``（官方推荐），
        故先剥 compile（``_orig_mod``）再剥 DDP（``module``），保存/加载统一走
        内层原始模型，保证 checkpoint 在 compile / DDP 与否之间均可互换。
        """
        model = getattr(self.model, '_orig_mod', self.model)
        return getattr(model, 'module', model)

    def _aggregate_loss(self, total_loss, total_samples):
        """训练/验证/测试 epoch 末尾的损失聚合钩子。

        单卡场景直接返回局部值；DDPTrainer 覆写为跨 rank all_reduce(SUM)，
        所有 rank 完成后返回同一 (global_loss, global_samples)，下游
        `if total_samples == 0` 检查天然跨 rank 一致，避免 OOM 全空时
        部分 rank 抛异常退出而其他 rank 卡在 metric.compute() 的 all_reduce 上死锁。

        Args:
            total_loss:    张量，本 rank 累积的加权损失（loss.detach() * batch_size 求和）
            total_samples: 本 rank 处理的样本数

        Returns:
            (global_loss, global_samples) —— 单卡时原样返回
        """
        return total_loss, total_samples

    def _autocast(self):
        """统一的 autocast 上下文：未启用 AMP 时 enabled=False，与 fp32 路径等价"""
        return autocast(device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp)

    def _load_state_dict(self, state_dict: Dict[str, Any], path: str) -> None:
        """
        加载权重并显式汇报匹配情况。

        去除 torch.compile 产生的 ``_orig_mod.`` 前缀（兼容旧 checkpoint）；
        用 strict=False 以容忍头/辅助分支的差异，但完全对不上时必须报错：
        否则会“加载成功”却拿着随机初始化的权重继续训练或评测。

        Args:
            state_dict: 待加载的权重字典
            path: checkpoint 路径（仅用于日志/报错信息）
        """
        # 去除 torch.compile 产生的 _orig_mod. 前缀（兼容旧 checkpoint）
        state_dict = {
            (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
            for k, v in state_dict.items()
        }
        target = self._unwrap_model()
        incompatible = target.load_state_dict(state_dict, strict=False)
        missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
        model_keys = set(target.state_dict().keys())
        loaded_count = len(model_keys) - len(missing)
        if loaded_count == 0:
            raise RuntimeError(
                f"从 {path} 未加载到任何权重：模型的 {len(model_keys)} 个键全部缺失"
                f"（checkpoint 提供 {len(state_dict)} 个键），权重与模型结构不匹配"
            )
        if missing or unexpected:
            self.logger.warning(
                f"⚠️ Partial state_dict load: {loaded_count}/{len(model_keys)} keys | "
                f"missing: {len(missing)} | unexpected: {len(unexpected)}"
            )
            if missing:
                self.logger.warning(f"   • Missing keys (first 5): {missing[:5]}")
            if unexpected:
                self.logger.warning(f"   • Unexpected keys (first 5): {unexpected[:5]}")


    def _restore_history(self, src_log: Path) -> None:
        """
        断点续训时衔接训练历史（曲线数据由 History 组件以 CSV 持久化）。

        从旧运行目录的 metrics.csv 读取记录，按 checkpoint 的 epoch 截断
        （丢弃比权重快照更晚的"幽灵"记录，如崩溃前多写的日志行、
        或从较早的 best.pt 恢复时其之后的记录），迁移到本次运行的新 CSV 文件，
        并重建内存中的绘图列表，使续训后绘制的曲线完整衔接。

        Args:
            src_log: checkpoint 同目录的 metrics.csv 路径，
                     不存在时保持空历史（仅绘制续训段）
        """
        if not src_log.exists():
            self.logger.warning(
                "⚠️ No metrics.csv found next to checkpoint, "
                "history curves start fresh"
            )
            return
        if src_log.resolve() == self.history.log_path.resolve():
            return  # 同一文件无需迁移（输出目录按时间戳隔离，正常不会发生）

        # 旧 CSV 的 fieldnames 仅用于 DictReader 的键约束；实际列名以文件表头为准
        old = History(src_log, fieldnames=self._build_history_fieldnames())
        old.load()
        records = [r for r in old.records if r.get('epoch', 0) <= self.start_epoch]
        for record in records:
            self.history.append(record)

        # 重建内存绘图列表（与 fit 循环中的追加逻辑对齐，列名采用 slash 前缀）
        def pick(phase: str, key: str) -> list:
            return [r[key] for r in records if r.get('phase') == phase and key in r]

        self.train_loss_all = pick('train', 'train/loss')
        self.val_loss_all = pick('val', 'val/loss')
        self.lr_history = pick('train', 'train/lr')
        self.val_epochs = pick('val', 'epoch')

        # 重建 val_metrics_history：扫描所有 'val/' 前缀的列，按列名收集
        # （列名集合由 _build_history_fieldnames 在 init_settings 时确定，
        #   这里只需遍历 val 记录里出现的所有 val/* 键）
        self.val_metrics_history = {}
        val_records = [r for r in records if r.get('phase') == 'val']
        if val_records:
            # 从第一条 val 记录收集所有 'val/' 前缀键（CSV 表头已统一，
            # 后续记录键集合一致；缺失值由 CSV 的空字符串表示，已转为 None）
            val_keys = [k for k in val_records[0].keys() if k.startswith('val/')]
            for k in val_keys:
                self.val_metrics_history[k] = [
                    r[k] for r in val_records if k in r and r[k] is not None
                ]
        self.logger.info(
            f"📈 Training history restored from {src_log}: "
            f"{len(self.train_loss_all)} train epochs, "
            f"{len(self.val_epochs)} validation rounds, "
            f"{len(self.val_metrics_history)} val metrics"
        )


    def _restore_tensorboard(self, src_tb_dir: Path) -> None:
        """
        断点续训时迁移 TensorBoard 事件文件到本次运行的日志目录。

        将旧运行目录的 tensorboard/ 事件文件复制到当前 save_dir/tensorboard/，
        使续训后 TensorBoard 能看到完整的历史曲线。SummaryWriter 检测到同名文件
        后会自动创建新的 event 文件（后缀 +1），不会覆盖或冲突。

        Args:
            src_tb_dir: 旧运行目录的 tensorboard/ 子目录路径，
                        不存在或为空时保持空历史（仅记录续训段）
        """
        if not src_tb_dir.exists() or not src_tb_dir.is_dir():
            self.logger.warning(
                "⚠️ No tensorboard dir found next to checkpoint, "
                "TensorBoard curves start fresh"
            )
            return

        dst_tb_dir = self.save_dir / 'tensorboard'
        if src_tb_dir.resolve() == dst_tb_dir.resolve():
            return  # 同一目录无需复制（输出目录按时间戳隔离，正常不会发生）

        event_files = list(src_tb_dir.glob('events.out.tfevents.*'))
        if not event_files:
            self.logger.warning(
                f"⚠️ No event files found in {src_tb_dir}, "
                "TensorBoard curves start fresh"
            )
            return

        os.makedirs(dst_tb_dir, exist_ok=True)
        copied = 0
        for src_file in event_files:
            dst_file = dst_tb_dir / src_file.name
            if dst_file.exists():
                continue  # 已存在同名文件，跳过（避免重复复制）
            shutil.copy2(src_file, dst_file)
            copied += 1

        self.logger.info(
            f"📈 TensorBoard events restored from {src_tb_dir}: "
            f"{copied} file(s) copied"
        )


    def _restore_best_weights(self) -> None:
        """训练结束后恢复 best.pt 权重，使后续 test() 评估基于最佳模型而非末轮。

        早停只负责“何时停”，不保证退出时内存中就是最佳权重；若不恢复，
        触发早停那一轮的较差权重会污染后续 test() 报告。
        best.pt 不存在时（如全程未触发 best 保存）保留末轮权重并告警。
        """
        best_path = self.save_dir / 'best.pt'
        if best_path.exists():
            self.logger.info("📥 Restoring best.pt for subsequent evaluation...")
            self.load_checkpoint(str(best_path))
        else:
            self.logger.warning("⚠️ best.pt not found, keeping last-epoch weights")

    def _log_final_hparams(self) -> None:
        """训练结束记录超参 + 最终指标到 TensorBoard HPARAMS 面板。

        'hparam/' 前缀为业内惯例，与普通标量曲线区分；指标取 best 与 final 两组，
        并从最后一次验证的完整结果中补充 f1/kappa/balanced_acc 等标量，
        使跨实验对比信息更完整。
        """
        final_metrics = {}
        if self.best_metric not in (float('inf'), float('-inf')):
            final_metrics[f'hparam/best_{self.monitor}'] = self.best_metric
        if self.val_loss_all:
            final_metrics['hparam/final_val_loss'] = self.val_loss_all[-1]
        val_acc_all = self.val_metrics_history.get('val/acc', [])
        if val_acc_all:
            final_metrics['hparam/final_val_acc'] = val_acc_all[-1]
        if self.val_metrics_result:
            for k, v in self.val_metrics_result.items():
                if k in ('val/loss', 'val/acc', 'val/time'):
                    continue  # 已在上文记录
                scalar = v.item() if hasattr(v, 'item') else v
                if isinstance(scalar, (int, float)):
                    final_metrics[f'hparam/final_{k}'] = scalar
        if final_metrics:
            self.tb_logger.log_hparams(self, final_metrics)


    ######### 核心训练组件 #########

    def fit(self) -> None:
        """
        执行完整训练流程。

        每个 epoch：train_epoch() 训练 → 按 eval_interval 调用 validation_epoch() 验证
        （最后一轮强制验证）→ 调度器 step → 训练历史落盘（CSV）→
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
                monitor_name=self.monitor,
                verbose=False,
            )
            self.logger.info(
                f"⏳ Early stopping enabled | monitor: {self.monitor} ({self.monitor_mode}) | "
                f"patience: {self.early_stop_patience} validation rounds "
                f"(≈ {self.early_stop_patience * self.eval_interval} epochs) | "
                f"delta: {self.early_stop_delta}"
                + (f" | min_epochs: {self.min_epochs}" if self.min_epochs > 0 else "")
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
            # 恢复早停器状态（load_checkpoint 已将 state 缓存到 self._early_stopper_state）
            if early_stopper is not None and self._early_stopper_state is not None:
                early_stopper.load_state_dict(self._early_stopper_state)
                self.logger.info(
                    f"🔄 EarlyStopping state restored | "
                    f"best {self.monitor}: {early_stopper.best_value} "
                    f"(counter={early_stopper.counter})"
                )
        if self.start_epoch >= self.max_epochs:
            self.logger.warning(
                f"⚠️ start_epoch ({self.start_epoch}) >= epochs ({self.max_epochs}), "
                f"skipping training loop"
            )

        # 标准日志：训练开始时记录一次模型结构图
        if self.train_loader is not None:
            sample, _ = next(iter(self.train_loader))
            self.tb_logger.log_graph(self._unwrap_model(), sample.to(self.device))

        for epoch in range(self.start_epoch, self.max_epochs):

            self.current_epoch = epoch + 1
            self.logger.info(f"📅 Epoch {self.current_epoch} / {self.max_epochs}")
            
            # 训练
            train_results = self.train_epoch()

            # 验证：每 eval_interval 轮一次；最后一轮强制验证，确保 best.pt 能覆盖末期模型
            should_validate = self.val_loader is not None and (
                self.current_epoch % self.eval_interval == 0
                or self.current_epoch == self.max_epochs)
            if should_validate:
                val_metrics = self.validation_epoch()
                # 统一 monitor 必须存在于验证指标中（首轮验证即 fail fast）
                # _resolve_monitor_key 找不到时会抛 KeyError，含 acc↔oa 兜底
                self._resolve_monitor_key(val_metrics)
            else:
                val_metrics = None
                # self.logger.warning("⚠️ No validation loader or no-validate epoch, skipping validation")

            # 调整学习率（Plateau 类调度器仅在有验证结果的轮次 step）
            current_lr = self._step_scheduler(val_metrics)
            self.lr_history.append(current_lr)

            # 训练历史落盘（CSV 追加，断点续训时据此衔接曲线）
            # 列名采用 Lightning 风格 slash 前缀；train/val 行字段互不重叠，
            # 缺失列由 History 填空字符串
            self.history.append({
                'epoch': self.current_epoch, 'phase': 'train',
                'train/loss': train_results['loss'],
                'train/lr': current_lr,
                'train/time': train_results['time'],
            })
            if val_metrics is not None:
                # val_metrics 键统一为 val/ 前缀风格，直接写入 CSV
                val_row = {
                    'epoch': self.current_epoch, 'phase': 'val',
                    'val/loss': val_metrics['val/loss'],
                    'val/acc': val_metrics['val/acc'],
                    'val/time': val_metrics['val/time'],
                }
                # 其余指标键已含 val/ 前缀，直接写入
                for k, v in val_metrics.items():
                    if k in ('val/loss', 'val/acc', 'val/time'):
                        continue
                    if isinstance(v, (int, float)) or (hasattr(v, 'ndim') and v.ndim == 0):
                        val_row[k] = v
                self.history.append(val_row)

            # ========== ✅ 验证轮次专属：更新最佳模型 + 早停判断 ==========
            # （先于 last.pt 保存，保证 last.pt 记录的 best_metric 是最新值）
            should_stop = False
            if val_metrics is not None:
                # 统一 monitor 取值（slash 前缀风格归一化到 val_metrics 原始键）
                _, monitored = self._resolve_monitor_key(val_metrics)
                if self._is_better(monitored):
                    self.best_metric = monitored
                    best_checkpoint = {
                        'monitor': self.monitor,
                        'best_metric': self.best_metric,
                        'val/acc': val_metrics['val/acc'],
                        'val/loss': val_metrics['val/loss'],
                        'epoch': self.current_epoch,
                        'state_dict': self._unwrap_model().state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                        'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
                        'config': {  # ✅ 额外保存配置，方便复现
                            'optimizer': type(self.optimizer).__name__ if self.optimizer else None,
                            'scheduler': type(self.scheduler).__name__ if self.scheduler else None,
                        }
                    }

                    # 保存固定文件名 best.pt
                    self.save_checkpoint('best.pt', checkpoint=best_checkpoint)

                    self.logger.info(
                        f"✨ New best model saved! | "
                        f"Epoch: {self.current_epoch} | "
                        f"{self.monitor}: {monitored:.4f} | "
                        f"Val Loss: {val_metrics['val/loss']:.4f}"
                    )

                # 早停检查（监控与 best.pt 相同的 monitor；仅判断是否继续训练，
                # eval_interval > 1 时 patience 按“验证次数”而非 epoch 数计）
                if early_stopper is not None:
                    early_stopper(
                        value=monitored,
                        epoch=self.current_epoch)
                    if early_stopper.early_stop:
                        if self.current_epoch < self.min_epochs:
                            # 保护期内抑制早停：重置计数器，避免保护期结束后立即触发
                            early_stopper.counter = 0
                            early_stopper.early_stop = False
                            self.logger.debug(
                                f"⏳ Early stopping suppressed at epoch {self.current_epoch} "
                                f"(min_epochs={self.min_epochs} protection active)"
                            )
                        else:
                            self.logger.info(
                                f"🛑 Early stopping triggered at epoch {self.current_epoch} "
                                f"(no {self.monitor} improvement for "
                                f"{early_stopper.patience} validation rounds)"
                            )
                            should_stop = True

            # ========== ✅ 保存最新模型 (last.pt) ==========
            last_checkpoint = {
                'epoch': self.current_epoch,
                'state_dict': self._unwrap_model().state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
                'val/loss': val_metrics['val/loss'] if val_metrics else None,
                'val/acc': val_metrics['val/acc'] if val_metrics else None,
                'monitor': self.monitor,
                'best_metric': self.best_metric,
                'train_loss': train_results['loss'],
                'early_stopper': early_stopper.state_dict() if early_stopper is not None else None,
            }
            self.save_checkpoint('last.pt', checkpoint=last_checkpoint)

            if should_stop:
                break

        # 训练结束前恢复最佳权重（主流做法：早停只管停训，后续评估用最佳模型）
        # 若不恢复，早停退出时内存中是触发轮次的较差权重，后续 test() 报告会失真
        self._restore_best_weights()

        # 可视化训练曲线（绘图逻辑见 trainers/visualizer.py，训练数据显式传入）
        self.visualizer_plot()

        # 标准日志：超参 + 最终指标对照表（供 HPARAMS 面板跨实验对比）
        self._log_final_hparams()

        # 关闭 TensorBoard 日志组件（由 init_settings 创建，生命周期随训练结束）
        self.tb_logger.close()

        # 关闭历史记录文件句柄（已逐条 flush，此处仅显式释放资源）
        self.history.close()

        # 释放 train/val loader 的 persistent workers，避免脚本退出时挂起；
        # test_loader 若有 persistent workers 也会一起清，test() 会重新 spawn
        self.cleanup()

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
                    desc=f'Epoch {self.current_epoch}/{self.max_epochs} [Train]', 
                    leave=False,
                    disable=self.pbar_disable,
                    )

        # for batch_idx, (inputs, targets) in enumerate(pbar):
        for batch_idx, batch in enumerate(pbar):

            try:
                # 前向（AMP 启用时在 autocast 下以低精度计算）+ 反向传播
                self.optimizer.zero_grad(set_to_none=True) 
                with self._autocast():
                    loss, batch_size = self.training_step(batch)
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
                total_loss += loss.detach() * batch_size  # 加权累加
                total_samples += batch_size

                # 批次级日志（每 log_interval 个 batch：进度条 + batch 级标准日志，
                # 共用一次 .item() 同步；低频写入也避免 event 文件过大）
                if batch_idx % self.log_interval == 0:
                    batch_loss = loss.item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'lr': f'{current_lr:.6e}'
                    })
                    self.tb_logger.log_scalars({'batch_loss': batch_loss, 'batch_lr': current_lr},
                                               self.global_step, 'train/')
            except RuntimeError as e:
                # 异常处理：跳过问题 batch，记录日志
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at batch {batch_idx}, skipping...")
                    self._empty_cache()
                    # 清掉失败尝试可能残留的部分梯度（如 backward 中途 OOM），
                    # set_to_none=True 比置零额外省一点显存
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e

        # 计算平均损失（全部 batch 因 OOM 被跳过时给出明确报错，而非除零崩溃）
        # DDP 下先跨 rank 聚合，保证所有 rank 对“是否全空”判定一致，避免死锁
        total_loss, total_samples = self._aggregate_loss(total_loss, total_samples)
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed in this training epoch "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        avg_loss = total_loss.item() / total_samples  # 加权平均更准确
        self.train_loss_all.append(avg_loss)

        # 记录训练元数据（epoch 末重新取 lr，避免 batch 级调度器下的过期值）
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time
        # 标准日志（epoch 级）
        self.tb_logger.log_scalars({'epoch_loss': avg_loss,
                                    'learning_rate': current_lr, 
                                    'samples_per_sec': samples_per_sec,}, 
                                    self.current_epoch, 'train/')

        # 日志
        self.logger.info(
            f"🏃 Train | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Speed: {samples_per_sec:.0f} samples/sec"
        )

        return  {'loss': avg_loss, 'time': epoch_time}


    def training_step(self, batch) -> torch.Tensor:
        """
        单个 batch 的前向推理 + 损失计算（不含反向传播）。

        子类可覆写此方法实现自定义训练逻辑
        （如多输出模型、多任务损失、深监督等）。

        Args:
            batch: 包括 输入张量inputs 和 真实标签targets

        Returns:
            标量损失张量（需保留计算图供 backward） 和 真实batch_size
        """
        inputs, targets = batch
        batch_size = inputs.size(0)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        # 分割标签常为 (N,1,H,W)，仅去掉通道维得到 (N,H,W)；
        # 分类标签 (N,) 不受影响；不用无参 squeeze()，
        # 避免 bs=1 时误删 batch 维（CE loss 会报 batch 不匹配）
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        targets = targets.long()
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return loss, batch_size


    @torch.no_grad()
    def validation_epoch(self) -> Dict[str, Any]:
        """
        在验证集上评估模型。
        前向推理、损失计算与指标累积委托给 validation_step（子类可覆写）。

        Returns:
            验证结果字典，键统一采用 'val/' 前缀风格：
            metrics.compute() 全部指标 + 'val/time'（验证耗时）
            如 'val/acc'、'val/f1'、'val/time'，供 monitor 直接匹配
        """

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics = self._val_metric  # 切换活跃指标为 val
        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(self.val_loader, 
                    desc=f'Epoch {self.current_epoch}/{self.max_epochs} [Valid]', 
                    leave=False,
                    disable=self.pbar_disable)

        for batch_idx, batch in enumerate(pbar):
            try:
                # 前向推理 + 损失 + 指标累积（见 validation_step；AMP 下同样用 autocast 加速）
                with self._autocast():
                    loss, _, batch_size = self.validation_step(batch) # inputs, targets

                # 加权累加损失（张量累加，避免每 batch 同步）
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
        # DDP 下先跨 rank 聚合损失与样本数，保证所有 rank 对“是否全空”判定一致
        total_loss, total_samples = self._aggregate_loss(total_loss, total_samples)
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed during validation "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        results = self.metrics.compute()
        # 将 loss 纳入 results，统一 val/ 前缀风格
        avg_loss = total_loss.item() / total_samples
        self.val_loss_all.append(avg_loss)
        results['val/loss'] = avg_loss
        val_acc = results.get('val/acc', results.get('val/pixel_acc', 0.0))

        # 记录元数据
        val_time = time.time() - start_time
        samples_per_sec = total_samples / val_time

        # 标准日志：汇总指标 → TensorBoard 标量曲线
        self.tb_logger.log_val_metrics(results, self.current_epoch)
        # 逐类指标 → TensorBoard val_per_class/ 分组（按需，仅当指标类支持时）
        if hasattr(self.metrics, 'per_class_metrics') and self.tb_logger.enabled:
            per_class = self.metrics.per_class_metrics()
            for k, v in per_class.items():
                scalar = v.item() if hasattr(v, 'item') else v
                self.tb_logger.writer.add_scalar(f'val_per_class/{k}', scalar, self.current_epoch)
        self.tb_logger.log_scalars({'epoch_acc': val_acc,
                                    'samples_per_sec': samples_per_sec,
                                    }, self.current_epoch, 'val/')
        
        # 更新历史 dict（val_epochs 记录对应轮次，eval_interval > 1 时绘图对齐用）
        # 所有指标统一存入 val_metrics_history，键名与 metrics.csv 列名一致
        self._append_val_metrics(results, val_time)
        self.val_epochs.append(self.current_epoch)
        self.val_metrics_result = results  # 保留详细结果供后续分析
        
        self.logger.info(
            f"🔍 Valid | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {val_acc:.4f} | " #TODO 可配置为monitor
            f"Speed: {samples_per_sec:.0f} samples/sec"
        )
        
        # 可选：记录详细指标到 debug 日志
        # self.logger.debug(f"Validation metrics detail: {results}")
        # validation_epoch 返回键统一为 val/ 前缀，直接合并 time 即可
        results['val/time'] = val_time
        return results


    def validation_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单个 batch 的评估逻辑：前向推理 + 损失计算 + 指标累积。
        验证（validation_epoch）与测试（test）共用此钩子，
        子类覆写一处即可同时生效（如多输出模型、自定义指标更新）。

        Args:
            batch

        Returns:
            (标量损失张量, logits)；多输出模型覆写时返回主输出 logits，
            供 test() 保存预测结果（argmax / 置信度）使用
        """
        inputs, targets = batch
        batch_size = inputs.size(0)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        # 分割标签常为 (N,1,H,W)，仅去掉通道维得到 (N,H,W)；
        # 分类标签 (N,) 不受影响；不用无参 squeeze()，
        # 避免 bs=1 时误删 batch 维（CE loss 会报 batch 不匹配）
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        logits = self.model(inputs)
        loss = self.criterion(logits, targets)

        # 指标 state 已 .to(self.device)，直接在训练设备上累积；
        # update 内部会 argmax 并对齐到 matrix 设备，无需手动搬运。
        # DDP 下 state 留在通信设备是 torchmetrics gather 同步的前提。
        self.metrics.update(logits.detach(), targets.detach())
        return loss, logits, batch_size


    @torch.no_grad()
    def test(self, report_results: bool = True,
             save_error_index: bool = False,
             save_predictions: bool = False) -> Optional[Dict[str, Any]]:
        """
        在测试集上评估模型（与 fit() 解耦，由用户在训练结束后手动调用）
        
        Args:
            report_results: 是否打印详细测试报告
            save_error_index: 是否保存错误索引
            save_predictions: 是否保存预测结果
        
        Returns:
            测试结果字典，包含：
                - loss, acc, time, samples, cnf_matrix 基础字段
                - metrics: compute() 完整结果（含 f1/kappa/balanced_acc 等全部指标）
            test_loader 未提供时直接返回 None
        """
        if self.test_loader is None:
            self.logger.warning("⚠️ No test dataloader provided, skipping test.")
            return None

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics = self._test_metric  # 切换活跃指标为 test
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
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 前向推理 + 损失 + 指标累积（与验证共用 validation_step 钩子，
                # 子类覆写后 test 同步生效）
                with self._autocast():
                    loss, outputs, batch_size = self.validation_step(batch) # inputs, targets
                
                # 加权累加损失（张量累加，避免每 batch 同步）
                # batch_size = inputs.size(0)
                total_loss += loss.detach() * batch_size
                total_samples += batch_size

                # 保存预测结果（用于后续分析/提交）
                if save_predictions and predictions is not None:
                    # 记录: (global_index, prediction, target, confidence)
                    inputs, targets = batch
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
        # DDP 下先跨 rank 聚合损失与样本数，保证所有 rank 对“是否全空”判定一致
        total_loss, total_samples = self._aggregate_loss(total_loss, total_samples)
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed during testing "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        avg_loss = total_loss.item() / total_samples
        results = self.metrics.compute()
        test_acc = results.get('test/acc', results.get('test/pixel_acc', 0.0))
        # 混淆矩阵通过 property 获取（cm.matrix）；
        # DDP 模式下已由 DDPTrainer.test() 在 super() 前 all_reduce
        cnf_matrix = self._test_metric.confusion_matrix.cpu().numpy()
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
        
        # 打印详细测试报告（报告格式化见 trainers/report.py）
        if report_results:
            print_test_report(
                results, cnf_matrix, test_time, samples_per_sec,
                class_names=self.class_names,
                prefix='test/',
                is_classification=self.is_classification,
                save_path=str(self.save_dir / 'test_report.txt'),
                logger=self.logger,
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
        
        # 释放 test_loader 的 persistent workers，避免脚本退出时挂起
        self.cleanup()
        
        return {'loss': avg_loss, 
                'acc': test_acc, 
                'time': test_time, 
                'samples': total_samples, 
                'cnf_matrix': cnf_matrix,
                'metrics': results,
                }


    @torch.no_grad()
    def evaluate_epoch(self) -> Dict[str, Any]:
        """
        在测试集上评估模型。
        前向推理、损失计算与指标累积委托给 validation_step（子类可覆写）。

        Returns:
            验证结果字典，键统一采用 'val/' 前缀风格：
            metrics.compute() 全部指标 + 'val/time'（验证耗时）
            如 'val/acc'、'val/f1'、'val/time'，供 monitor 直接匹配
        """

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics = self._val_metric  # 切换活跃指标为 val
        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(self.val_loader, 
                    desc=f'Epoch {self.current_epoch}/{self.max_epochs} [Test]', 
                    leave=False,
                    disable=self.pbar_disable)

        for batch_idx, batch in enumerate(pbar):
            try:
                # 前向推理 + 损失 + 指标累积（见 validation_step；AMP 下同样用 autocast 加速）
                with self._autocast():
                    loss, _, batch_size = self.validation_step(batch) # inputs, targets

                # 加权累加损失（张量累加，避免每 batch 同步）
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
        # DDP 下先跨 rank 聚合损失与样本数，保证所有 rank 对“是否全空”判定一致
        total_loss, total_samples = self._aggregate_loss(total_loss, total_samples)
        if total_samples == 0:
            raise RuntimeError(
                "No samples processed during validation "
                "(all batches skipped, likely OOM). Try reducing batch size."
            )
        results = self.metrics.compute()
        # 将 loss 纳入 results，统一 val/ 前缀风格（分类取 val/acc，分割取 val/pixel_acc）
        avg_loss = total_loss.item() / total_samples
        results['val/loss'] = avg_loss
        val_acc = results.get('val/acc', results.get('val/pixel_acc', 0.0))

        # 记录元数据
        val_time = time.time() - start_time
        samples_per_sec = total_samples / val_time

        # 标准日志：汇总指标 → TensorBoard 标量曲线
        self.tb_logger.log_val_metrics(results, self.current_epoch)
        # 逐类指标 → TensorBoard val_per_class/ 分组（按需，仅当指标类支持时）
        if hasattr(self.metrics, 'per_class_metrics') and self.tb_logger.enabled:
            per_class = self.metrics.per_class_metrics()
            for k, v in per_class.items():
                scalar = v.item() if hasattr(v, 'item') else v
                self.tb_logger.writer.add_scalar(f'val_per_class/{k}', scalar, self.current_epoch)
        self.tb_logger.log_scalars({'epoch_acc': val_acc,
                                    'samples_per_sec': samples_per_sec,
                                    }, self.current_epoch, 'val/')
        
        # 更新历史 dict（val_epochs 记录对应轮次，eval_interval > 1 时绘图对齐用）
        # 所有指标统一存入 val_metrics_history，键名与 metrics.csv 列名一致
        self._append_val_metrics(results, val_time)
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

        # validation_epoch 返回键统一为 val/ 前缀，直接合并 time 即可
        results['val/time'] = val_time
        return results


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
        with self._autocast():
            logits = self.model(inputs)
        return torch.argmax(logits, dim=1)


    ########## 其他组件 #################


    def save_checkpoint(self, filename: str, checkpoint: Optional[Dict[str, Any]] = None) -> str:
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
        if filename == 'best.pt':
            self.logger.info(f"💾 Model saved: {filename}")
        return str(model_path)


    def load_checkpoint(self, checkpoint_fn: Optional[str] = None, resume: bool = False) -> str:
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

            # 兼容多种格式：完整 checkpoint 字典（model_state_dict / state_dict）/ 纯 state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self._load_state_dict(state_dict, checkpoint_fn)

            # 断点续训：恢复训练状态（仅加载权重时不动优化器/进度，
            # 避免最终评估恢复 best.pt 时污染训练状态）
            if resume and isinstance(checkpoint, dict):
                if 'optimizer' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

                if 'lr_schedule' in checkpoint and checkpoint['lr_schedule'] and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['lr_schedule'])
                    # 手动同步 optimizer 各参数组的 LR 到 last_epoch 对应值，
                    # 消除 resume 后首个 epoch 训练期间 LR 与调度器进度不一致的偏差
                    # （LambdaLR.state_dict 不含 lr_lambdas，load_state_dict 仅恢复
                    #  last_epoch / base_lrs，不会自动回写 optimizer.param_groups）
                    if hasattr(self.scheduler, 'lr_lambdas'):
                        last_epoch = self.scheduler.last_epoch
                        for pg, base_lr, lr_fn in zip(
                            self.optimizer.param_groups,
                            self.scheduler.base_lrs,
                            self.scheduler.lr_lambdas,
                        ):
                            pg['lr'] = base_lr * lr_fn(last_epoch)

                # 恢复 AMP 损失缩放器状态（仅 CUDA + AMP 训练的 checkpoint 才有）
                if checkpoint.get('scaler') and self.scaler.is_enabled():
                    self.scaler.load_state_dict(checkpoint['scaler'])

                # checkpoint 中 epoch 为已完成轮次，fit 从该索引继续（current_epoch = epoch + 1）
                self.start_epoch = checkpoint.get('epoch', 0) or 0
                # 恢复历史最佳监控值：优先取新格式 best_metric（last.pt 也会记录）；
                # 旧 checkpoint 仅有 best_val_acc / val_acc（acc 语义，仅 monitor='val/acc' 时可复用）
                ckpt_monitor = checkpoint.get('monitor')
                # 旧 ckpt 的 monitor 是 'loss'/'acc' 短名，归一化为新风格 'val/loss'/'val/acc'
                # 以便与当前 self.monitor（强制 slash 前缀）比较
                if ckpt_monitor is not None and '/' not in ckpt_monitor:
                    ckpt_monitor = f'val/{ckpt_monitor}'
                restored_best = checkpoint.get('best_metric')
                if restored_best is None and self.monitor == 'val/acc':
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

                # 恢复早停器状态（缓存到类属性，fit 中创建 early_stopper 后再还原）
                self._early_stopper_state = checkpoint.get('early_stopper')

                # 恢复训练历史曲线：从 checkpoint 同目录的 metrics.csv 迁移
                # （曲线数据由 History 组件持久化，不入 checkpoint）
                self._restore_history(Path(checkpoint_fn).parent / 'metrics.csv')

                # 恢复 TensorBoard 事件文件：从旧目录的 tensorboard/ 迁移
                # （使续训后 TensorBoard 能看到完整历史曲线）
                self._restore_tensorboard(Path(checkpoint_fn).parent / 'tensorboard')

                self.logger.info(
                    f"🔄 Training state restored | "
                    f"start_epoch: {self.start_epoch} | "
                    f"best {self.monitor}: {self.best_metric:.4f}"
                )

            # ✅ 加载时打印关键指标（兼容新旧 checkpoint 键名格式）
            epoch = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
            # 新格式 'val/acc'/'val/loss'，旧格式 'val_acc'/'val_loss'
            val_acc = None
            val_loss = None
            if isinstance(checkpoint, dict):
                val_acc = checkpoint.get('val/acc', checkpoint.get('val_acc'))
                val_loss = checkpoint.get('val/loss', checkpoint.get('val_loss'))
            
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
        loaded_path = self.load_checkpoint(checkpoint_fn, resume=True)
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


    def visualizer_plot(self):
        # 1. loss_curve：训练流程图（train loss + val loss 对比）
        if self.train_loss_all and self.val_loss_all:
            self.visualizer.plot_loss_curve(
                train_loss=self.train_loss_all,
                val_loss=self.val_loss_all,
                val_epochs=self.val_epochs,
                save_path=os.path.join(self.save_dir, 'loss_curve.png'),
            )
        # 2. val_metrics：验证指标图（单图多线，acc/f1/precision/recall 等）
        #    排除 val/loss（已在 loss_curve 中绘制）和 val/time（不属于质量指标）
        val_metrics_for_plot = {
            k: v for k, v in self.val_metrics_history.items()
            if k not in ('val/loss', 'val/time') and len(v) == len(self.val_epochs)
        }
        if val_metrics_for_plot and self.val_epochs:
            self.visualizer.plot_val_metrics(
                metrics=val_metrics_for_plot,
                val_epochs=self.val_epochs,
                save_path=os.path.join(self.save_dir, 'val_metrics.png'),
            )
        # 3. lr_curve：学习率曲线
        if self.lr_history:
            self.visualizer.plot_lr_history(
                self.lr_history,
                save_path=os.path.join(self.save_dir, 'lr_curve.png'),
            )


    def cleanup(self) -> None:
        """
        释放 DataLoader 的 persistent workers 与缓存，避免脚本退出时挂起。

        背景：num_workers > 0 且 persistent_workers=True 时，worker 子进程
        在 epoch 间不退出，只有 _DataLoaderIter 被 GC 时才会 _shutdown_workers()
        终止。若 trainer 持有 loader 引用未释放，Python GC 时机不确定，
        worker 可能存活至解释器退出，导致脚本 hang。

        做法：清空各 loader 的 _iterator 字段（_DataLoaderIter 引用），
        让 GC 回收时触发 worker shutdown；DataLoader 对象本身保持可用，
        下次 iter() 会重新 spawn workers。幂等，可重复调用。
        """
        for loader in (self.train_loader, self.val_loader, self.test_loader):
            if loader is None:
                continue
            # _iterator 由 iter(loader) 创建，persistent_workers=True 时跨 epoch 复用
            it = getattr(loader, '_iterator', None)
            if it is not None:
                # 显式 shutdown workers（兼容不同 PyTorch 版本：失败则走 GC 兜底）
                shutdown = getattr(it, '_shutdown_workers', None)
                if shutdown is not None:
                    try:
                        shutdown()
                    except Exception:
                        # 任意异常都忽略：worker 已退出或未启动时不影响主流程
                        pass
                loader._iterator = None
        gc.collect()
        self._empty_cache()



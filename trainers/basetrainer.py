
import os
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any, List, Tuple
import logging 

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from metrics import TorchMetricsWrapper

# 日志配置
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制：当验证损失不再改善时提前停止训练
    
    Attributes:
        patience: 容忍的 epoch 数，超过后停止训练
        verbose: 是否打印详细信息
        delta: 损失改善的最小阈值
        save_fn: 模型保存回调函数
    """
    def __init__(self,
                 patience: int = 10,
                 delta: float = 0.0,
                 save_fn: Optional[callable] = None,
                 verbose: bool = False,
                 ):
        self.patience = patience
        self.delta = delta
        self.save_fn = save_fn
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        检查是否需要早停
        
        Args:
            val_loss: 验证集损失
            model: 模型对象
            epoch: 当前 epoch
            metrics: 其他指标字典（可选）
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch, metrics)
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
            self._save_checkpoint(val_loss, model, epoch, metrics)
            self.counter = 0

    def _save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """保存最佳模型检查点"""
        if self.save_fn:
            checkpoint = {
                'val_loss': val_loss,
                'epoch': epoch,
                'model': model.state_dict(),
            }
            if metrics:
                checkpoint.update(metrics)
            self.save_fn(checkpoint)
        self.val_loss_min = val_loss


class BaseTrainer:
    """
    通用深度学习训练器，支持分类和分割任务
    
    核心功能:
        - 自动设备检测与分配
        - 灵活的优化器/调度器配置
        - 完整的训练/验证/测试流程
        - 早停机制和模型检查点
        - 丰富的日志和可视化输出
    """
    def __init__(self,
                 model: nn.Module=None,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 num_classes: int = 2,
                 epochs: int = 10,
                 optimizer_cfg: Optional[Dict[str, Any]] = None,
                 scheduler_cfg: Optional[Dict[str, Any]] = None,
                 criterion: nn.Module = nn.CrossEntropyLoss(),  # None 时自动使用 
                 device: str = 'auto',  # 'auto' | 'cuda' | 'cpu'
                 output_dir: str='./output',
                 resume: Optional[str]=None,
                 compile_model:bool = False, 
                 max_grad_norm: Optional[float] = None,  # 梯度裁剪
                 class_names: Optional[List[str]] = None,
                 is_classification: bool = True, # 是否为分类任务（影响指标计算和日志记录）
                 tensorboard_writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
                 **kwargs):
        """
        初始化训练器
        
        :param optimizer_cfg: 优化器配置字典
            示例: {"type": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "momentum": 0.9}
        :param scheduler_cfg: 调度器配置字典（None 表示不使用）
            示例: {"type": "reduceLROnPlateau", "mode": "min", "patience": 5, "factor": 0.5}
        """
        # 时间戳（用于输出目录命名）
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设备配置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        # 输出目录
        self.save_dir = Path(os.path.join(output_dir, self.timestamp))
        # 核心组件
        self.model = model.to(self.device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader or val_dataloader

        # 任务配置
        self.num_classes = num_classes
        self.epochs = epochs
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.class_names = class_names or [f'Class-{i}' for i in range(num_classes)]

        # 优化配置
        self.optimizer = None
        self.scheduler = None
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.max_grad_norm = max_grad_norm
        # 检查是否为 batch 级调度器（如 OneCycleLR）
        self.is_batch_scheduler = (
            scheduler_cfg and 
            scheduler_cfg.get("type", "").lower() == "onecyclelr"
        )

        # 恢复训练
        self.resume = resume
        # 编译选项
        self.compile_model = compile_model
        # 日志器
        self.logger = logger
        # 当前 epoch
        self.current_epoch = 0
        # 模型文件名
        self.model_name = None
        # TensorBoard writer（可选）
        self.writer = tensorboard_writer
        # 指标记录
        self.metrics = None
        self.train_loss_all = []
        self.val_loss_all = []
        self.train_acc_all = []
        self.val_acc_all = []
        self.lr_history = []
        self.cnf_matrix = None
        self.val_metrics_result = None
        
        self.is_classification = is_classification

        self.init_settings()


    def init_settings(self) -> None:
        """初始化训练环境"""
        self.logger.info("📋 Initializing training environment...")

        # 输出目录
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"📁 Output directory: {self.save_dir}")

        self.logger.info(f"🤖 Setting up device: {self.device}")

        # 优化器和调度器
        self.logger.info("🔧 Initializing optimizer and scheduler...")
        self.init_optim_scheduler(self.optimizer_cfg, self.scheduler_cfg)

        # 恢复训练
        if self.resume:
            self.logger.info(f"📥 Resuming from checkpoint: {self.resume}")
            self.load_model(self.resume)
            self.model_name = os.path.basename(self.resume)

        # 指标计算器
        self.logger.info("📊 Initializing metrics calculator...")
        self.metrics = TorchMetricsWrapper(self.num_classes, 'cpu') # 指标计算在 CPU 上进行，避免 GPU 内存占用过高
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
            scheduler_cfg: Optional[Dict[str, Any]] = None
            ) -> None:
        """
        初始化优化器和学习率调度器（支持配置字典 + 工厂模式） 
        :param optimizer_cfg: 优化器配置字典
        :param scheduler_cfg: 调度器配置字典
        """

        # 优化器工厂：类型 -> 类映射
        OPTIMIZER_FACTORY = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
            }

        # 调度器工厂
        SCHEDULER_FACTORY = {
            "steplr": lr_scheduler.StepLR,
            "exponentiallr": lr_scheduler.ExponentialLR,
            "reducelronplateau": lr_scheduler.ReduceLROnPlateau,
            "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
            "onecyclelr": lr_scheduler.OneCycleLR,
            }

        # ========== 优化器配置 ==========
        default_opt_cfg = {
            "type": "adam",
            "lr": 1e-4, # self.lr,
            "weight_decay": 1e-4 if optimizer_cfg and optimizer_cfg.get("type") == "adamw" else 0,
            "momentum": 0.9,  # 仅对 SGD 生效
            "betas": (0.9, 0.999),  # 仅对 Adam/AdamW 生效
            "eps": 1e-8,  # 仅对 Adam/AdamW 生效
        }
        # 合并配置：传入的 cfg 优先
        if optimizer_cfg:
            default_opt_cfg.update(optimizer_cfg)
        opt_cfg = default_opt_cfg

        opt_type = opt_cfg["type"].lower()
        if opt_type not in OPTIMIZER_FACTORY:
            raise ValueError(f"Unsupported optimizer: {opt_type}. Available: {list(OPTIMIZER_FACTORY.keys())}")
        
        # 提取优化器专用参数（避免传入无关参数报错）
        opt_class = OPTIMIZER_FACTORY[opt_type]
        opt_kwargs = {"lr": opt_cfg["lr"], "weight_decay": opt_cfg["weight_decay"]}
        if opt_type in ["adam", "adamw"]:
            opt_kwargs["betas"] = opt_cfg.get("betas", (0.9, 0.999))
            opt_kwargs["eps"] = opt_cfg.get("eps", 1e-8)
        elif opt_type == "sgd":
            opt_kwargs["momentum"] = opt_cfg.get("momentum", 0.9)
            opt_kwargs["nesterov"] = opt_cfg.get("nesterov", True)
        else:
            pass
            # raise ValueError(f"Unsupported optimizer type: {opt_type}")
        
        # 创建优化器（自动过滤 requires_grad=False 的参数）
        self.optimizer = opt_class(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            **opt_kwargs
        )

        # 记录日志
        self.logger.info(
            f"🎯 Optimizer: {opt_type.upper()} | "
            f"LR: {opt_cfg['lr']:.2e} | "
            f"Weight Decay: {opt_cfg['weight_decay']:.2e}"
        )

        # ========== 调度器配置 ==========
        if scheduler_cfg is None or scheduler_cfg.get("type", "").lower() in ["none", "null", ""]:
            self.scheduler = None
            self.logger.info("Scheduler: None (using constant learning rate)")
            return
        
        # 默认调度器配置
        default_sched_cfg = {
            "type": "steplr",
            "step_size": 10,
            "gamma": 0.1,
            "mode": "min",         # for ReduceLROnPlateau
            "patience": 5,         # for ReduceLROnPlateau
            "factor": 0.5,         # for ReduceLROnPlateau
            "T_max": self.epochs,  # for CosineAnnealingLR
            "eta_min": 1e-6,       # for CosineAnnealingLR
        }
        default_sched_cfg.update(scheduler_cfg)
        sched_cfg = default_sched_cfg

        sched_type = sched_cfg["type"].lower()
        if sched_type not in SCHEDULER_FACTORY:
            raise ValueError(f"Unsupported scheduler: {sched_type}. Available: {list(SCHEDULER_FACTORY.keys())}")
        
        sched_class = SCHEDULER_FACTORY[sched_type]
        
        # 提取调度器专用参数
        sched_kwargs = {}
        if sched_type == "steplr":
            sched_kwargs = {"step_size": sched_cfg["step_size"], 
                            "gamma": sched_cfg["gamma"]}
        elif sched_type == "exponentiallr":
            sched_kwargs = {"gamma": sched_cfg["gamma"]}
        elif sched_type == "reducelronplateau":
            sched_kwargs = {
                "mode": sched_cfg["mode"], 
                "factor": sched_cfg["factor"], 
                "patience": sched_cfg["patience"],
            }
        elif sched_type == "cosineannealinglr":
            sched_kwargs = {"T_max": sched_cfg["T_max"], 
                            "eta_min": sched_cfg["eta_min"]}
        elif sched_type == "onecyclelr":
            # OneCycleLR 需要额外参数，这里简化处理
            sched_kwargs = {
                "max_lr": sched_cfg.get("max_lr", opt_cfg["lr"] * 10),
                "epochs": self.epochs,
                "steps_per_epoch": len(self.train_loader),
            }
        
        # 创建调度器
        self.scheduler = sched_class(self.optimizer, **sched_kwargs)

        # 记录日志
        self.logger.info(f"Scheduler: {sched_type}, Config: {sched_kwargs}")


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
        
        # OneCycleLR 在 train_step 内已调用
        if self.is_batch_scheduler:
            return self.optimizer.param_groups[0]['lr']
        # 区分调度器类型
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            monitor_key = getattr(self.scheduler, 'monitor', None) or \
                         self.scheduler_cfg.get('monitor', 'loss')
            mode = getattr(self.scheduler, 'mode', 'min')
            
            metric = val_metrics.get(monitor_key)
            if metric is None:
                self.logger.warning(
                    f"Monitor key '{monitor_key}' not found in val_metrics, using 'loss'"
                )
                metric = val_metrics['loss']
            
            # ReduceLROnPlateau 默认找最小值
            if monitor_key == 'acc' and mode == 'min':
                metric = -metric
            
            self.scheduler.step(metric)
            self.logger.debug(
                f"ReduceLROnPlateau step: {monitor_key}={metric:.4f}"
            )
        else:
            self.scheduler.step()
        
        return self.optimizer.param_groups[0]['lr']


    def fit(self) -> None:
        """
        执行完整的训练流程,逐步优化模型，保存最佳表现的模型。
        训练循环: 对于每个 epoch：
                加载数据: 从 train_loader 中加载一个批次的图像和对应的标签，将其移动到指定设备上（如 GPU）。
                清零梯度: 在每次反向传播之前，调用 self.optimizer.zero_grad() 清除上一次计算的梯度，避免累积。
                前向传播: 将输入图像通过模型，得到预测输出。
                计算损失: 使用损失函数 criterion 计算预测输出和真实标签之间的误差。
                反向传播: 调用 loss.backward() 计算损失对模型参数的梯度。
                优化模型参数: 调用 self.optimizer.step() 更新模型参数，最小化损失。
                累积损失: 记录当前批次的损失值，以便后续计算平均训练损失。
                设置模型为训练模式: 调用 self.model.train()，确保模型在训练过程中正确处理 dropout 和 batch normalization 等操作。
                批次循环: 对每个数据批次：
                调整学习率: 在每个 epoch 完成后，调用 self.scheduler.step() 依据预设的策略调整学习率。
                计算平均训练损失: 通过累积的损失计算该 epoch 的平均训练损失，并记录。
                验证模型: 调用 evaluate 方法，使用验证集评估模型的表现。
                保存模型: 如果当前 epoch 的验证精度超过历史最佳精度，则保存该 epoch 的模型，并更新最佳精度记录。
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

        save_fn = lambda ckpt: self.save_model(f"best.pt",checkpoint=ckpt)
        early_stopper = EarlyStopping(
            patience=self.scheduler_cfg.get("patience", 5) if self.scheduler_cfg else 5,
            save_fn=save_fn,
            verbose=True,
        )

        best_val_acc = 0.0
        for epoch in range(self.epochs):

            self.current_epoch = epoch + 1
            self.logger.info(f"📅 Epoch {self.current_epoch}/{self.epochs}")
            
            # 训练
            train_results = self.train_step()
            # 验证
            if self.val_loader is not None:
                val_results = self.evaluate()
            else:
                self.logger.warning("⚠️ No validation loader, skipping validation")
                val_results = {'loss': 0.0, 'acc': 0.0, 'time': 0.0}
            # 调整学习率
            current_lr = self._step_scheduler(val_results)
            self.lr_history.append(current_lr)

            # # 调度器调整学习率（关键！区分类型）
            # if self.scheduler is not None and not self.is_batch_scheduler:
            #     if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            #         # ReduceLROnPlateau 需要手动传入监控指标
            #         monitor_key = self.scheduler_cfg.get("monitor", "loss")  # 'loss' or 'acc'
            #         metric = val_results[monitor_key]
            #         # 如果监控准确率，需要取负（因为 ReduceLROnPlateau 默认找最小值）
            #         if monitor_key == "acc":
            #             metric = -metric  # 或设置 mode='max'
            #         self.scheduler.step(metric)
            #         current_lr = self.optimizer.param_groups[0]['lr']
            #         self.logger.info(f"LR adjusted by ReduceLROnPlateau: {current_lr:.2e} (metric: {monitor_key}={val_results[monitor_key]:.4f})")
            #     else:
            #         # 其他调度器：按 epoch 自动调整
            #         self.scheduler.step()
            #         current_lr = self.optimizer.param_groups[0]['lr']
            #         self.logger.info(f"LR adjusted by {type(self.scheduler).__name__}: {current_lr:.2e}")
            # else:
            #     # 无调度器：记录当前固定学习率
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     self.logger.debug(f"LR (fixed): {current_lr:.2e}")

            # ========== ✅ 保存最新模型 (last.pt) ==========
            last_checkpoint = {
                'epoch': self.current_epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_results['loss'],
                'val_acc': val_results['acc'],
                'train_loss': train_results['loss'],
                'train_acc': train_results['acc'],
            }
            self.save_model('last.pt', checkpoint=last_checkpoint)

            # ========== ✅ 保存最佳模型 (best.pt + 详细文件名) ==========
            val_acc = val_results['acc']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint = {
                    'val_acc': best_val_acc,
                    'epoch': self.current_epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                    'config': {  # ✅ 额外保存配置，方便复现
                        'optimizer_cfg': self.optimizer_cfg,
                        'scheduler_cfg': self.scheduler_cfg,
                    }
                }
                
                # 保存固定文件名 best.pt（方便加载）
                self.save_model('best.pt', checkpoint=best_checkpoint)
                
                # 保存带指标的详细文件名（用于归档）
                # detailed_name = f"epoch_{self.current_epoch}_acc_{val_acc:.4f}.pt"
                # self.save_model(detailed_name, checkpoint=best_checkpoint)
                
                self.logger.info(
                    f"✨ New best model saved! | "
                    f"Epoch: {self.current_epoch} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val Loss: {val_results['loss']:.4f}"
            )
            
            # 早停检查
            early_stopper(
                model=self.model,
                val_loss=val_results['loss'], 
                epoch=self.current_epoch,
                metrics={'val_acc': val_acc})
            if early_stopper.early_stop:
                self.logger.info("🛑 Early stopping triggered")
                break

            # 定期测试
            if (self.current_epoch) % 10 == 0:
                test_results = self.test(report_results=False)
                self.logger.info(
                    f"🧪 Test | "
                    f"Loss: {test_results['loss']:.4f} | "
                    f"Acc: {test_results['acc']:.4f}"
                )

        # 最终测试
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎯 Running final test...")
        final_test = self.test(report_results=True, save_predictions=True)
        self.cnf_matrix = final_test['cnf_matrix']
        # self.logger.info(f"Test Loss: {final_test['loss']:.4f} | Acc: {final_test['acc']:.4f}")
        # 可视化
        self.plot_acc_loss(save_path=os.path.join(self.save_dir, 'acc_loss.png'))
        if self.lr_history:
            self.plot_lr_history(save_path=os.path.join(self.save_dir, 'lr_curve.png'))
        if self.is_classification and self.cnf_matrix is not None:
            self.plot_confusion_matrix(
                cm=self.cnf_matrix,
                class_names=self.class_names,
                normalize=False,
                save_path=self.save_dir / 'confusion_matrix.png'
            )
            self.plot_confusion_matrix(
                cm=self.cnf_matrix,
                class_names=self.class_names,
                normalize=True,
                save_path=self.save_dir / 'confusion_matrix_normalized.png'
            )


    def train_step(self) -> Dict[str, Any]:
        """
        执行一个 epoch 的训练
        
        Returns:
            训练结果字典 {'loss', 'acc', 'time'}
        """
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics.reset()
        self.model.train()

        # 记录当前学习率（用于日志）
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.current_epoch == 1:
            self.logger.info("start training ...")
        pbar = tqdm(self.train_loader, 
                    desc=f'Epoch {self.current_epoch}/Train', 
                    leave=False)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # # 调试模式：打印第一个 batch 的统计信息
                # if self.current_epoch == 0 and batch_idx == 0:
                #     print(f"[DEBUG] Input shape: {inputs.shape}, Range: [{inputs.min():.3f}, {inputs.max():.3f}]")
                #     print(f"[DEBUG] Targets: {targets[:10].tolist()}")
                #     print(f"[DEBUG] Outputs shape: {outputs.shape}, Requires grad: {outputs.requires_grad}")

                # 前向 + 反向传播
                self.optimizer.zero_grad(set_to_none=True) 
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                # 梯度裁剪（防止爆炸，可选）
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm)

                self.optimizer.step()

                # ✅ OneCycleLR 需要在 batch 后调用 step()
                if self.scheduler is not None and self.is_batch_scheduler:
                    self.scheduler.step()

                # 指标累积
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size  # 加权累加
                total_samples += batch_size

                preds = torch.argmax(outputs, dim=1).detach().cpu()
                # convert to CPU！！！
                targets_cpu = targets.detach().cpu()  # 也转 CPU，避免设备不匹配
                self.metrics.update(preds, targets_cpu)

                # 批次级日志（每 50 个 batch 更新一次进度条）
                if batch_idx % 50 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
            except RuntimeError as e:
                # 异常处理：跳过问题 batch，记录日志
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # 计算汇总指标
        avg_loss = total_loss / total_samples  # 加权平均更准确
        results = self.metrics.compute()
        train_acc = results.get('total_acc', 0.0)

        # 记录训练元数据
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time
        # 记录到 TensorBoard（如果启用）
        if self.writer is not None:
            self.writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)
            self.writer.add_scalar("train/epoch_acc", train_acc, self.current_epoch)
            self.writer.add_scalar("train/learning_rate", current_lr, self.current_epoch)
            self.writer.add_scalar("train/samples_per_sec", samples_per_sec, self.current_epoch)
            
        self.train_loss_all.append(avg_loss)
        self.train_acc_all.append(train_acc)

        # 日志
        self.logger.info(
            f"🏃 Train | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {train_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Speed: {samples_per_sec:.0f} samples/sec"
        )

        return  {'loss': avg_loss, 
                 'acc': train_acc, 
                 'time': epoch_time}

 
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        在验证集上评估模型
        
        Returns:
            验证结果字典 {'loss', 'acc', 'time'}
        """

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics.reset()
        self.model.eval()

        # self.logger.info("🧪 Running evaluation...")
        pbar = tqdm(self.val_loader, 
                    desc=f'Epoch {self.current_epoch}/Valid', 
                    leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 加权累加损失
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(outputs, dim=1).detach().cpu()
                targets_cpu = targets.detach().cpu()
                self.metrics.update(preds, targets_cpu)
                
                # 进度条实时更新
                if batch_idx % 20 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            except RuntimeError as e:
                # 异常处理：跳过问题 batch
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at val batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # 计算汇总指标
        avg_loss = total_loss / total_samples
        results = self.metrics.compute()
        val_acc = results['total_acc']

        # 记录元数据
        val_time = time.time() - start_time
        samples_per_sec = total_samples / val_time

        # TensorBoard 记录（如果启用）
        if self.writer is not None:
            self.writer.add_scalar("val/epoch_loss", avg_loss, self.current_epoch)
            self.writer.add_scalar("val/epoch_acc", val_acc, self.current_epoch)
            self.writer.add_scalar("val/samples_per_sec", samples_per_sec, self.current_epoch)
        
        # 更新历史列表
        self.val_loss_all.append(avg_loss)
        self.val_acc_all.append(val_acc)
        self.val_metrics_result = results  # 保留详细结果供后续分析
        
        self.logger.info(
            f"🔍 Valid | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Speed: {samples_per_sec:.0f} samples/sec"
        )
        
        # 可选：记录详细指标到 debug 日志
        # self.logger.debug(f"Validation metrics detail: {results}")

        # avg_loss = total_loss /self.val_count #len(self.train_loader.dataset)
        # 计算并记录指标
        # results = self.metrics.compute()
        # self.val_metrics_result = results
        # self.logger.info("Validation metrics:")
        # for key, value in results.items():
        #     self.logger.info(f"{key}: {value}")
        # val_acc = results['total_acc']
        # self.val_acc_all.append(val_acc)
        # self.val_loss_all.append(avg_loss)
        return {'loss': avg_loss, 
                'acc': val_acc, 
                'time': val_time}


    @torch.no_grad()
    def test(self, 
             report_results: bool = True,
             save_predictions: bool = False,
             ) -> Dict[str, Any]:
        """
        在测试集上评估模型
        
        Args:
            save_predictions: 是否保存预测结果
            save_path: 预测结果保存路径
        
        Returns:
            测试结果字典 {'loss', 'acc', 'time', 'samples', 'cnf_matrix'}
        """
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.model.eval()
        self.metrics.reset()

        # 用于保存预测结果（如果需要）
        predictions = [] if save_predictions else None
        
        # self.logger.info("🎯 Testing model...")
        pbar = tqdm(self.test_loader, 
                    desc='Testing', 
                    leave=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 加权累加损失
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # 指标更新（转 CPU 避免显存泄漏）
                preds = torch.argmax(outputs, dim=1).detach().cpu()
                targets_cpu = targets.detach().cpu()
                self.metrics.update(preds, targets_cpu)

                # 保存预测结果（用于后续分析/提交）
                if save_predictions and predictions is not None:
                    # 记录: (global_index, prediction, target, confidence)
                    confidences = torch.softmax(outputs, dim=1).max(dim=1).values.detach().cpu()
                    for i, (p, t, c) in enumerate(zip(preds, targets_cpu, confidences)):
                        global_idx = batch_idx * self.test_loader.batch_size + i
                        predictions.append({
                            'index': global_idx,
                            'pred': p.item(),
                            'target': t.item(),
                            'confidence': c.item(),
                            'correct': p.item() == t.item()
                        })
                
                # 进度条实时更新
                if batch_idx % 20 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
            except RuntimeError as e:
                # 异常处理：跳过问题 batch
                if "out of memory" in str(e):
                    print(f"⚠️ OOM at test batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # 计算汇总指标
        avg_loss = total_loss / total_samples
        results = self.metrics.compute()
        test_acc = results.get('total_acc', 0.0)
        cnf_matrix = results.get('confmat')

        test_time = time.time() - start_time
        samples_per_sec = total_samples / test_time

        # 保存预测结果（如果需要）
        if save_predictions and predictions:
            import pandas as pd
            save_path = os.path.join(self.save_dir, 'test_predictions.csv')
            df = pd.DataFrame(predictions)
            df.to_csv(save_path, index=False)
            print(f"📁 Predictions saved to {save_path}")
            
            # 可选：保存错误样本索引（用于错误分析）
            errors = [p['index'] for p in predictions if not p['correct']]
            if errors:
                error_path = os.path.join(self.save_dir, 'test_errors.txt')
                with open(error_path, 'w') as f:
                    f.write('\n'.join(map(str, errors)))
                print(f"❌ {len(errors)} errors logged to {error_path}")
        
        # 打印详细测试报告
        if report_results:
            self._print_test_report(results, test_time, samples_per_sec)
        
        return {# **results,  #  展开 metrics 的所有指标 (total_acc, confmat, etc.)
                'loss': avg_loss, 
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
            inputs: 输入张量
        
        Returns:
            预测类别标签
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if inputs.dim() == 3:  # (C, H, W)
            inputs = inputs.unsqueeze(0)  # (1, C, H, W)
        inputs = inputs.to(self.device)
        self.model.eval()
        pred = self.model(inputs)
        pred = torch.argmax(pred, dim=1)
        return pred


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
        
        # 保存文件
        if checkpoint:
            torch.save(checkpoint, model_path)
        else:
            # 仅保存模型参数（用于轻量级部署）
            torch.save(self.model.state_dict(), model_path)
        
        self.logger.info(f"💾 Model saved: {filename}")
        return str(model_path)


    def load_model(self, checkpoint_fn: Optional[str] = None) -> str:
        """
        加载模型检查点
        
        Args:
            checkpoint_fn: 检查点文件路径
                        如果为 None，则自动按优先级查找: last.pt → best.pt
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
            checkpoint = torch.load(checkpoint_fn, weights_only=False, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model'], strict=False)
            
            if 'optimizer' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'lr_schedule' in checkpoint and checkpoint['lr_schedule'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['lr_schedule'])
            
            # ✅ 加载时打印关键指标（替代文件名中的信息）
            epoch = checkpoint.get('epoch', 'N/A')
            val_acc = checkpoint.get('val_acc')
            val_loss = checkpoint.get('val_loss')
            
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
        loaded_path = self.load_model(checkpoint_fn)
        self.logger.info(f"🔄 Resuming training from {loaded_path}")
        
        # 继续执行训练（从当前 epoch 开始）
        self.fit()


    def export_onnx(self, output_path: str = 'model.onnx', opset_version: int = 11) -> None:
        self.model.eval()

        # 创建一个示例输入张量
        # 替换 (1, 3, 224, 224) 为你模型的实际输入尺寸
        x, y = next(iter(self.train_loader))
        input_shape = list(x[0,:].cpu().numpy().shape)
        input_shape.insert(0, 1)
        dummy_input = torch.randn(input_shape).to(self.device)

        # 导出模型
        torch.onnx.export(
            self.model,                # 要转换的模型
            dummy_input,               # 示例输入张量
            output_path,               # 输出的 ONNX 文件名
            input_names=['input'],     # 输入节点名称（可选）
            output_names=['output'],   # 输出节点名称（可选）
            opset_version=opset_version # ONNX 操作集版本（通常使用最新支持的版本）
        )

        print(f"✅ Model exported to ONNX: {output_path}")



    def plot_lr_history(self, save_path: str = None):
        """绘制学习率变化曲线"""
        if not hasattr(self, 'lr_history') or not self.lr_history:
            return
        
        plt.figure(figsize=(8, 4))
        plt.plot(self.lr_history, 'bo-', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate (log scale)')
        plt.yscale('log')  # 对数坐标更清晰
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"📈 LR curve saved to {save_path}")
        # plt.show()
        plt.close()


    def _print_test_report(
        self,
        results: Dict[str, Any],
        elapsed_time: float,
        speed: float
    ) -> None:
        """打印格式化的测试报告"""
        print("\n" + "=" * 60)
        print("🧪 TEST REPORT".center(60))
        print("=" * 60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  • Accuracy     : {self._fmt(results.get('total_acc'), '.2f', scale=100, suffix='%')}")
        # print(f"  • Loss         : {self._fmt(results.get('loss'))}")
        if results.get('kappa') is not None:
            print(f"  • Kappa        : {self._fmt(results['kappa'])}")
        if results.get('total_iou') is not None:
            print(f"  • Mean IoU     : {self._fmt(results['total_iou'], '.2f', scale=100, suffix='%')}")
        
        # 每类指标
        if 'acc' in results and isinstance(results['acc'], list) and len(results['acc']) == self.num_classes:
            print(f"\n📋 Per-Class Accuracy:")
            for cls_idx, acc in enumerate(results['acc']):
                cls_name = self.class_names[cls_idx] if hasattr(self, 'class_names') else f"Class-{cls_idx}"
                print(f"  • {cls_name:12s}: {self._fmt(acc, '.2f', scale=100, suffix='%')}")
        
        # 性能指标
        print(f"\n⚡ Performance:")
        print(f"  • Samples      : {results.get('samples', 'N/A')}")
        print(f"  • Time         : {self._fmt(elapsed_time, '.2f', suffix='s')}")
        print(f"  • Speed        : {self._fmt(speed, '.0f', suffix=' samples/sec')}")
        # 混淆矩阵摘要
        if hasattr(self.metrics, 'confmat') and self.metrics.confmat is not None:
            cfm = self.metrics.confmat.compute()
            print(f"\n🔍 Confusion Matrix Summary:")
            print(f"  • Diagonal (correct)   : {torch.diag(cfm).sum().item()}")
            print(f"  • Off-diagonal (error) : {cfm.sum().item() - torch.diag(cfm).sum().item()}")
        
        print("=" * 60 + "\n")




    def plot_acc_loss(self, save_path=None):
        """绘制训练/验证损失和准确率曲线"""
        if not (self.train_loss_all and self.val_loss_all and self.train_acc_all and self.val_acc_all):
            raise ValueError("One or more of the data lists is empty or None.")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_all, 'ro-', label='Train Loss')
        plt.plot(self.val_loss_all, 'bs-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_all, 'ro-', label='Train Acc')
        plt.plot(self.val_acc_all, 'bs-', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()
        plt.close()


    def plot_lr_history(self, save_path: Optional[str] = None) -> None:
        """绘制学习率变化曲线"""
        if not self.lr_history:
            return
        
        plt.figure(figsize=(8, 4))
        plt.plot(self.lr_history, 'bo-', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate (log scale)')
        plt.yscale('log')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"📈 LR curve saved to {save_path}")
        
        plt.close()


    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        cmap: str = 'Blues',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        fontsize: int = 10,
        show_values: bool = True,
        value_format: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵 (num_classes × num_classes)
            class_names: 类别名称列表
            normalize: 是否按行归一化
            title: 图表标题
            cmap: 颜色映射
            save_path: 保存路径
            figsize: 画布大小
            fontsize: 字体大小
            show_values: 是否显示数值
            value_format: 数值格式
        
        Returns:
            matplotlib Figure 对象
        """
        cm_display = cm.copy()
        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_display = cm_display.astype('float') / cm_display.sum(axis=1, keepdims=True)
                cm_display = np.nan_to_num(cm_display)
        
        if value_format is None:
            value_format = '.1%' if normalize else '.0f'
        
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        im = ax.pcolormesh(cm_display, cmap=cmap, edgecolors='white', linewidths=0.5)
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks + 0.5)
        ax.set_yticks(tick_marks + 0.5)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=fontsize)
        ax.set_yticklabels(class_names, fontsize=fontsize)
        ax.invert_yaxis()
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count' if not normalize else 'Proportion', rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=fontsize + 2, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=fontsize)
        ax.set_ylabel('True Label', fontsize=fontsize)

        if show_values:
            thresh = cm_display.max() / 2.0 if not normalize else 0.5
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = cm_display[i, j]
                    text = f"{val:{value_format}}"
                    color = 'white' if val > thresh else 'black'
                    ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                           color=color, fontsize=fontsize - 2)
        
        if len(class_names) <= 20:
            for i in range(len(class_names)):
                rect = plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='gold', linewidth=2, alpha=0.5)
                ax.add_patch(rect)
        
        plt.tight_layout()
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📁 Confusion matrix saved to {save_path}")
        
        return fig


    def _fmt(
        self,
        val: Optional[float],
        pattern: str = ".4f",
        default: str = "N/A",
        scale: float = 1.0,
        suffix: str = ""
    ) -> str:
        """
        安全格式化数值
        
        Args:
            val: 数值
            pattern: 格式化模式
            default: 默认值（当 val 为 None 时）
            scale: 缩放因子
            suffix: 后缀字符串
        
        Returns:
            格式化后的字符串
        """
        if val is None:
            return default
        try:
            return f"{val * scale:{pattern}}{suffix}"
        except (TypeError, ValueError):
            return default

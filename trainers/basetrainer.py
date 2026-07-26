
import os
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Any, List
import logging 

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from metrics import Metrics
from optimizers import build_optimizer, build_scheduler, clip_grad_norm
from .visualizer import TrainingVisualizer
from .utils import EarlyStopping
from .logger import get_logger, add_file_handler


# 日志配置
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
                 model: nn.Module=None,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 num_classes: int = 2,
                 epochs: int = 10,
                 log_interval: int = 5,
                 eval_interval: int = 1,  # 每隔多少个 epoch 验证一次（1 = 每轮都验证）
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
                 tensorboard_writer: Optional[Any] = None,  # torch.utils.tensorboard.SummaryWriter
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
        self.class_names = class_names or [f'Class-{i}' for i in range(num_classes)]
        self.epochs = epochs
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # 优化配置
        self.optimizer = None
        self.scheduler = None
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.max_grad_norm = max_grad_norm
        # 是否为 batch 级调度器（如 OneCycleLR），在 init_optim_scheduler 中按实例类型判定
        self.is_batch_scheduler = False

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
        # 可视化器：只持有展示配置（输出目录/类别名），训练数据由调用时显式传入
        self.visualizer = TrainingVisualizer(
            save_dir=self.save_dir,
            class_names=self.class_names,
            logger=self.logger,
        )
        # 指标记录
        self.metrics = None
        self.train_loss_all = []
        self.val_loss_all = []
        self.val_acc_all = []
        self.val_epochs = []  # 记录每次验证对应的 epoch（eval_interval > 1 时绘图用）
        self.lr_history = []
        self.cnf_matrix = None
        self.val_metrics_result = None
        
        self.is_classification = is_classification
        # self.epoch = 0
        self.global_step = 0
        self.log_interval = log_interval
        self.eval_interval = max(1, eval_interval)

        self.init_settings()


    def init_settings(self) -> None:
        """初始化训练环境"""
        self.logger.info("📋 Initializing training environment...")

        # 输出目录
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"📁 Output directory: {self.save_dir}")
        add_file_handler(self.logger, self.save_dir/ "train.log")


        self.logger.info(f"🤖 Setting up device: {self.device}")

        # 优化器和调度器
        self.logger.info("🔧 Initializing optimizer and scheduler...")
        self.init_optim_scheduler(self.optimizer_cfg, self.scheduler_cfg)

        # 恢复训练
        if self.resume:
            self.logger.info(f"📥 Resuming from checkpoint: {self.resume}")
            self.load_model(self.resume)
            self.model_name = os.path.basename(self.resume)

        # 指标计算器（基于混淆矩阵，在 CPU 上累积，避免 GPU 内存占用过高）
        self.logger.info("📊 Initializing metrics calculator...")
        if self.metrics is None:
            # 分类任务不忽略任何标签；分割任务默认忽略 255（未标注区域）
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
            scheduler_cfg: Optional[Dict[str, Any]] = None
            ) -> None:
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

        # 早停器：只负责停训判断，best.pt 的保存由下方训练循环按 val_acc 统一管理
        early_stopper = EarlyStopping(
            patience=self.scheduler_cfg.get("patience", 5) if self.scheduler_cfg else 5,
            verbose=False,
        )

        best_val_acc = 0.0
        for epoch in range(self.epochs):

            self.current_epoch = epoch + 1
            self.logger.info(f"📅 Epoch {self.current_epoch}/{self.epochs}")
            
            # 训练
            train_results = self.train_epoch()

            # 验证：每 eval_interval 轮一次；最后一轮强制验证，确保 best.pt 能覆盖末期模型
            should_validate = self.val_loader is not None and (
                self.current_epoch % self.eval_interval == 0
                or self.current_epoch == self.epochs
            )
            if should_validate:
                val_metrics = self.evaluate_epoch()
            else:
                val_metrics = None
                if self.val_loader is None:
                    self.logger.warning("⚠️ No validation loader, skipping validation")

            # 调整学习率（Plateau 类调度器仅在有验证结果的轮次 step）
            current_lr = self._step_scheduler(val_metrics)
            self.lr_history.append(current_lr)

            # ========== ✅ 保存最新模型 (last.pt) ==========
            last_checkpoint = {
                'epoch': self.current_epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_metrics['loss'] if val_metrics else None,
                'val_acc': val_metrics['acc'] if val_metrics else None,
                'train_loss': train_results['loss'],
            }
            self.save_model('last.pt', checkpoint=last_checkpoint)

            # ========== ✅ 验证轮次专属：保存最佳模型 + 早停判断 ==========
            if val_metrics is not None:
                val_acc = val_metrics['acc']
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

                    self.logger.info(
                        f"✨ New best model saved! | "
                        f"Epoch: {self.current_epoch} | "
                        f"Val Acc: {val_acc:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f}"
                    )

                # 早停检查（仅判断是否继续训练，不保存模型；
                # eval_interval > 1 时 patience 按“验证次数”而非 epoch 数计）
                early_stopper(
                    val_loss=val_metrics['loss'], 
                    epoch=self.current_epoch)
                if early_stopper.early_stop:
                    self.logger.info("🛑 Early stopping triggered")
                    break

        # 最终测试前恢复最佳权重（主流做法：早停只管停训，评估用最佳模型）
        # 若不恢复，早停退出时内存中是触发轮次的较差权重，测试报告会失真
        best_path = self.save_dir / 'best.pt'
        if best_path.exists():
            self.logger.info("📥 Restoring best.pt for final evaluation...")
            self.load_model(str(best_path))
        else:
            self.logger.warning("⚠️ best.pt not found, evaluating with last-epoch weights")

        # 最终测试
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎯 Running final test...")
        final_test = self.test(report_results=True, save_predictions=True)
        self.cnf_matrix = final_test['cnf_matrix']
        # 可视化（绘图逻辑见 trainers/visualizer.py，训练数据显式传入）
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
        if self.is_classification and self.cnf_matrix is not None:
            self.visualizer.plot_confusion_matrix(
                cm=self.cnf_matrix,
                normalize=False,
                save_path=self.save_dir / 'confusion_matrix.png'
            )
            self.visualizer.plot_confusion_matrix(
                cm=self.cnf_matrix,
                normalize=True,
                save_path=self.save_dir / 'confusion_matrix_normalized.png'
            )


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
                    # leave=False
                    )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 前向 + 反向传播（前向推理与损失计算见 training_step）
                self.optimizer.zero_grad(set_to_none=True) 
                loss = self.training_step(inputs, targets)
                loss.backward()

                # 梯度裁剪（防止爆炸，可选）
                if self.max_grad_norm is not None:
                    clip_grad_norm(self.model, self.max_grad_norm)

                self.optimizer.step()
                self.global_step += 1

                # ✅ OneCycleLR 需要在 batch 后调用 step()
                if self.scheduler is not None and self.is_batch_scheduler:
                    self.scheduler.step()

                # 损失累积
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size  # 加权累加
                total_samples += batch_size

                # 批次级日志（每 log_interval 个 batch 更新一次进度条）
                if batch_idx % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
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

        # 计算平均损失
        avg_loss = total_loss / total_samples  # 加权平均更准确

        # 记录训练元数据（epoch 末重新取 lr，避免 batch 级调度器下的过期值）
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time
        # 记录到 TensorBoard（如果启用）
        if self.writer is not None:
            self.writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)
            self.writer.add_scalar("train/learning_rate", current_lr, self.current_epoch)
            self.writer.add_scalar("train/samples_per_sec", samples_per_sec, self.current_epoch)
            
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
            验证结果字典 {'loss', 'acc', 'time'}
        """

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(self.val_loader, 
                    desc=f'Epoch {self.current_epoch}/{self.epochs} [Valid]', 
                    # leave=False
                    )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 前向推理 + 损失 + 指标累积（见 validation_step）
                loss = self.validation_step(inputs, targets)

                # 加权累加损失
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # 进度条实时更新
                if batch_idx % self.log_interval == 0:
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
        val_acc = results['oa']  # OA: Overall Accuracy

        # 记录元数据
        val_time = time.time() - start_time
        samples_per_sec = total_samples / val_time

        # TensorBoard 记录（如果启用）
        if self.writer is not None:
            self.writer.add_scalar("val/epoch_loss", avg_loss, self.current_epoch)
            self.writer.add_scalar("val/epoch_acc", val_acc, self.current_epoch)
            self.writer.add_scalar("val/samples_per_sec", samples_per_sec, self.current_epoch)
        
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

        return {'loss': avg_loss, 
                'acc': val_acc, 
                'time': val_time}


    def validation_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        单个 batch 的验证逻辑：前向推理 + 损失计算 + 指标累积。

        子类可覆写此方法实现自定义验证逻辑（如多输出模型、自定义指标更新）。

        Args:
            inputs: 输入张量（已在目标设备上）
            targets: 真实标签（已在目标设备上）

        Returns:
            标量损失张量
        """
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        # Metrics 的混淆矩阵在 CPU 上，先搬运避免 GPU 训练时设备不匹配
        # （logits 传入后由 Metrics.update 自动 argmax）
        self.metrics.update(logits.detach().cpu(), targets.detach().cpu())
        return loss


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
        test_acc = results.get('oa', 0.0)
        # 混淆矩阵从指标计算器直接获取，转 numpy 供绘图使用
        cnf_matrix = self.metrics.confusion_matrix.cpu().numpy()

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
        
        # 打印详细测试报告（报告格式化见 trainers/visualizer.py）
        if report_results:
            self.visualizer.print_test_report(
                results, cnf_matrix, test_time, samples_per_sec,
                is_classification=self.is_classification,
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

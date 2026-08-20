"""
TensorBoard 日志组件：封装 SummaryWriter 的生命周期与写入接口。

设计原则：
    - 统一收拢所有 SummaryWriter 调用（标量曲线 / 模型结构 / 超参对比），
      按业内标准工具的形式记录，训练主流程（fit / train_epoch / evaluate_epoch）
      无需感知 TensorBoard 的存在；
    - 未启用时所有写入方法静默 no-op，调用方无需判空；
    - 后续迁移 wandb / mlflow 等工具时，只需实现一个接口相同的替代类，
      Trainer 侧代码无需改动；
    - 可脱离 Trainer 独立使用（如在 notebook 中复盘已有 event 文件）。

使用示例：
    tb = TensorBoardLogger(log_dir='./output/tensorboard', enabled=True)
    tb.log_scalars({'loss': 0.5, 'lr': 1e-3}, step=1, prefix='train/')
    tb.log_graph(model, sample_input)
    tb.log_hparams(trainer)
    tb.close()
"""

import logging
from typing import Dict, Optional, Any

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    TensorBoard 日志组件，封装 SummaryWriter 的生命周期与写入接口。

    职责：
        - Writer 创建 / 关闭（生命周期管理）
        - 标量、模型图、超参数的写入（统一入口）
        - 未启用时所有方法静默 no-op，调用方无需判空

    Args:
        log_dir: 日志目录（通常为 save_dir/tensorboard）
        enabled: 是否启用（False 时 writer 不创建，所有写入方法空操作）
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.writer: Optional[SummaryWriter] = None
        if enabled:
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"📈 TensorBoard logging to: {log_dir}")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """是否已启用（writer 已创建）。"""
        return self.writer is not None

    # ------------------------------------------------------------------
    # 写入方法
    # ------------------------------------------------------------------

    def log_scalars(self,
                    scalars: Dict[str, Any],
                    step: int,
                    prefix: str = '') -> None:
        """
        批量记录标量曲线。未启用时静默跳过；非数值项自动忽略。

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
        记录 metrics.compute() 的全部验证指标。

        新指标体系下键名已含 'val/' 前缀（如 'val/acc'），直接写入 TensorBoard，
        不再额外加 prefix；非标量值（混淆矩阵等）自动过滤。
        """
        if self.writer is None:
            return
        scalars = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                scalars[k] = v
            elif hasattr(v, 'ndim') and v.ndim == 0:
                scalars[k] = v.item()
        self.log_scalars(scalars, step, '')  # 空前缀，键名已含 val/

    def log_graph(self, model, sample_input) -> None:
        """
        记录模型结构图（训练开始时调用一次；失败仅告警不阻断训练）。

        add_graph 基于 trace，部分动态控制流模型 / 特殊设备可能失败，
        因此包在 try 中。

        Args:
            model: 模型实例（建议先 _unwrap_model 去掉 DDP / compile 包装）
            sample_input: 用于 trace 的样本输入
        """
        if self.writer is None:
            return
        try:
            self.writer.add_graph(model, sample_input)
            logger.info("📈 Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log model graph: {e}")

    def log_hparams(self, trainer, metrics: Dict[str, float]) -> None:
        """
        记录超参数与最终指标的对照表（训练结束时调用一次，
        供 TensorBoard HPARAMS 面板跨实验对比）。

        从 trainer 实例中提取基础超参（模型名、优化器、调度器等），
        与最终指标一起写入。add_hparams 仅支持 int/float/str/bool，
        其余类型自动转 str。

        Args:
            trainer: BaseTrainer 实例（用于提取超参）
            metrics: 最终指标字典，键名建议带 'hparam/' 前缀（业内惯例，
                     与普通标量曲线区分）
        """
        if self.writer is None:
            return
        # 基础超参（add_hparams 仅支持 int/float/str/bool，其余类型转 str）
        hparams: Dict[str, Any] = {
            'model': type(trainer._unwrap_model()).__name__,
            'criterion': type(trainer.criterion).__name__,
            'epochs': trainer.max_epochs,
            'use_amp': trainer.use_amp,
            'monitor': trainer.monitor,
        }
        batch_size = getattr(trainer.train_loader, 'batch_size', None)
        if batch_size is not None:
            hparams['batch_size'] = batch_size
        if trainer.max_grad_norm is not None:
            hparams['max_grad_norm'] = trainer.max_grad_norm
        # 优化器/调度器超参（从实例提取，避免依赖外部 cfg 字典）
        if trainer.optimizer is not None:
            hparams['optim/type'] = type(trainer.optimizer).__name__
            hparams['optim/lr'] = trainer.optimizer.param_groups[0]['lr']
            hparams['optim/weight_decay'] = trainer.optimizer.param_groups[0]['weight_decay']
            hparams['optim/param_groups'] = len(trainer.optimizer.param_groups)
        if trainer.scheduler is not None:
            hparams['sched/type'] = type(trainer.scheduler).__name__
        try:
            # run_name='.' 写入当前 run 目录，避免产生额外的时间戳子 run
            self.writer.add_hparams(hparams, metrics, run_name='.')
            logger.info("📈 Hparams logged to TensorBoard")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log hparams: {e}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭 writer，释放资源。未启用时静默跳过。"""
        if self.writer is not None:
            self.writer.close()


# 向后兼容别名
TBLogger = TensorBoardLogger

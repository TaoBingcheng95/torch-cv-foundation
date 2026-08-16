"""
训练可视化模块：训练曲线、学习率曲线、混淆矩阵绘制。

设计原则：
    - 构造器只持有"展示配置"（输出目录、类别名、日志器），一次配置、处处复用；
    - 训练数据（损失历史、混淆矩阵、指标字典）由方法参数显式传入，
      不持有 Trainer 引用，保持 Trainer -> Visualizer 的单向依赖；
    - 因此本类可脱离训练器独立使用（如在 notebook 中复盘实验数据）。
    - 文本报告生成（print_test_report）已迁移至 trainers/report.py。

使用示例：
    viz = Visualizer(save_dir='./output', class_names=['cat', 'dog'])
    viz.plot_loss_curve(train_loss, val_loss, val_epochs,
                        save_path='./output/loss_curve.png')
    viz.plot_val_metrics({'val/acc': acc_list, 'val/f1': f1_list},
                         val_epochs, save_path='./output/val_metrics.png')
    viz.plot_confusion_matrix(cm, normalize=True, save_path='./output/cm.png')
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _ema_smooth(values: Sequence[float], alpha: float) -> List[float]:
    """
    指数移动平均（EMA）平滑。

    Args:
        values: 原始数值序列
        alpha: 平滑系数（0~1），越大越平滑

    Returns:
        平滑后的等长列表
    """
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * ema[-1] + (1 - alpha) * v)
    return ema


class Visualizer:
    """
    训练图表可视化器（可脱离训练器独立使用）。

    所有 plot 方法均返回 matplotlib Figure 对象，便于调用方进一步定制或
    在 notebook 中内联展示。

    Args:
        save_dir: 默认输出目录（各 plot 方法未显式传 save_path 时使用）
        class_names: 类别名称列表（混淆矩阵使用）
        logger: 日志器（None 时使用模块级 logger）
    """

    def __init__(self,
                 save_dir: Optional[Union[str, Path]] = None,
                 class_names: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        self.class_names = class_names or []
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _resolve_path(self, save_path, default_name: str) -> Optional[str]:
        """save_path 为 None 且配置了 save_dir 时，落到 save_dir/default_name。"""
        if save_path is not None:
            return str(save_path)
        if self.save_dir is not None:
            return str(self.save_dir / default_name)
        return None

    def _save_and_close(self, fig: plt.Figure, save_path: Optional[str],
                        log_msg: str) -> None:
        """统一处理图片保存与 figure 关闭。"""
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            self.logger.info(log_msg)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 绘图方法
    # ------------------------------------------------------------------

    def plot_loss_curve(self,
                       train_loss: Sequence[float],
                       val_loss: Sequence[float],
                       val_epochs: Optional[Sequence[int]] = None,
                       save_path: Optional[str] = None,
                       smooth: Optional[float] = None,
                       figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        """
        绘制训练/验证损失对比曲线（单图双线）。

        Args:
            train_loss: 每个 epoch 的训练损失
            val_loss: 每次验证的损失
            val_epochs: 每次验证对应的 epoch（eval_interval > 1 时横轴对齐用；
                        None 时按 1..len(val_loss) 顺序排布）
            save_path: 保存路径（None 时使用 save_dir/loss_curve.png）
            smooth: 可选 EMA 平滑系数（0~1），None 不平滑。
                    平滑后原始数据用淡色底线保留
            figsize: 画布大小

        Returns:
            matplotlib Figure 对象
        """
        if not (train_loss and val_loss):
            raise ValueError("Loss history is empty, run fit() first.")
        if val_epochs is None:
            val_epochs = list(range(1, len(val_loss) + 1))

        train_epochs = list(range(1, len(train_loss) + 1))
        fig, ax = plt.subplots(figsize=figsize)

        # 训练损失
        if smooth and 0 < smooth < 1:
            ax.plot(train_epochs, train_loss, color='red', alpha=0.25, linewidth=1)
            ax.plot(train_epochs, _ema_smooth(train_loss, smooth),
                    color='red', linestyle='-', linewidth=1.8, label='Train Loss')
        else:
            ax.plot(train_epochs, train_loss, 'ro-', label='Train Loss', linewidth=1.5)

        # 验证损失
        if smooth and 0 < smooth < 1:
            ax.plot(val_epochs, val_loss, color='blue', alpha=0.25, linewidth=1)
            ax.plot(val_epochs, _ema_smooth(val_loss, smooth),
                    color='blue', linestyle='-', linewidth=1.8, label='Val Loss')
        else:
            ax.plot(val_epochs, val_loss, 'bs-', label='Val Loss', linewidth=1.5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train and Validation Loss')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        save_path = self._resolve_path(save_path, 'loss_curve.png')
        self._save_and_close(fig, save_path, f"📈 Loss curve saved to {save_path}")
        return fig

    def plot_val_metrics(self,
                        metrics: Dict[str, Sequence[float]],
                        val_epochs: Sequence[int],
                        save_path: Optional[str] = None,
                        smooth: Optional[float] = None,
                        figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        """
        绘制验证指标曲线（单图多线，共享 X/Y 轴）。

        所有验证指标值域通常在 [0, 1]，共享 Y 轴便于横向对比趋势。
        每条线在末端标注当前值，图例用键名去掉 'val/' 前缀。

        Args:
            metrics: 指标名 → 数值列表。键名用 CSV 列名风格
                     （如 'val/acc'、'val/f1'、'val/iou'）
            val_epochs: 每次验证对应的 epoch
            save_path: 保存路径（None 时使用 save_dir/val_metrics.png）
            smooth: 可选 EMA 平滑系数（0~1），None 不平滑。
                    平滑后原始数据用淡色底线保留
            figsize: 画布大小

        Returns:
            matplotlib Figure 对象
        """
        if not metrics:
            raise ValueError("Metrics dict is empty, nothing to plot.")
        # 过滤掉空列表或长度与 val_epochs 不一致的指标
        valid = {k: list(v) for k, v in metrics.items()
                 if v and len(v) == len(val_epochs)}
        if not valid:
            raise ValueError("No valid metrics to plot (length mismatch or empty)")

        fig, ax = plt.subplots(figsize=figsize)
        # 线型循环：实线 / 虚线 / 点划线 / 点线，避免仅靠颜色区分
        line_styles = ['-', '--', '-.', ':']
        colors = plt.cm.tab10.colors  # 10 色循环

        for idx, (name, values) in enumerate(valid.items()):
            label = name.split('/')[-1]  # 去掉 'val/' 前缀
            color = colors[idx % len(colors)]
            ls = line_styles[idx % len(line_styles)]

            if smooth and 0 < smooth < 1:
                # EMA 平滑：smoothing factor = smooth
                ema = _ema_smooth(values, smooth)
                # 原始数据用淡色底线
                ax.plot(val_epochs, values, color=color, alpha=0.25, linewidth=1)
                ax.plot(val_epochs, ema, color=color, linestyle=ls,
                        linewidth=1.8, label=label)
            else:
                ax.plot(val_epochs, values, color=color, linestyle=ls,
                        linewidth=1.5, label=label)

            # 末端标注当前值
            if values:
                last_epoch = val_epochs[-1]
                last_val = values[-1]
                ax.annotate(f'{last_val:.3f}',
                            xy=(last_epoch, last_val),
                            xytext=(5, 0), textcoords='offset points',
                            fontsize=8, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Validation Metrics')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        save_path = self._resolve_path(save_path, 'val_metrics.png')
        self._save_and_close(fig, save_path,
                             f"📈 Validation metrics curve saved to {save_path}")
        return fig

    def plot_lr_history(self,
                        lr_history: Sequence[float],
                        save_path: Optional[str] = None,
                        figsize: Tuple[float, float] = (8, 4)) -> plt.Figure:
        """
        绘制学习率变化曲线（对数坐标）。

        Args:
            lr_history: 每个 epoch 的学习率
            save_path: 保存路径（None 时使用 save_dir/lr_curve.png）
            figsize: 画布大小

        Returns:
            matplotlib Figure 对象
        """
        if not lr_history:
            raise ValueError("LR history is empty, nothing to plot.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lr_history, 'bo-', label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate (log scale)')
        ax.set_yscale('log')  # 对数坐标更清晰
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        save_path = self._resolve_path(save_path, 'lr_curve.png')
        self._save_and_close(fig, save_path, f"📈 LR curve saved to {save_path}")
        return fig

    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              normalize: bool = False,
                              title: str = 'Confusion Matrix',
                              cmap: str = 'Blues',
                              save_path: Optional[str] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              fontsize: int = 10,
                              show_values: bool = True,
                              value_format: Optional[str] = None) -> plt.Figure:
        """
        绘制混淆矩阵

        Args:
            cm: 混淆矩阵 (num_classes × num_classes)
            class_names: 类别名称列表（None 时使用构造器配置）
            normalize: 是否按行归一化
            title: 图表标题
            cmap: 颜色映射
            save_path: 保存路径
            figsize: 画布大小（None 时根据类别数自动调整）
            fontsize: 字体大小
            show_values: 是否显示数值
            value_format: 数值格式

        Returns:
            matplotlib Figure 对象
        """
        class_names = class_names or self.class_names
        n_cls = len(class_names)

        # 根据类别数自动调整画布大小（≤10 类用默认尺寸，更多类别时线性放大）
        if figsize is None:
            base = max(6, n_cls * 0.6)
            figsize = (int(base), int(base * 0.8))

        cm_display = cm.copy()
        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_display = cm_display.astype('float') / cm_display.sum(axis=1, keepdims=True)
                cm_display = np.nan_to_num(cm_display)

        if value_format is None:
            value_format = '.1%' if normalize else '.0f'

        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        im = ax.pcolormesh(cm_display, cmap=cmap, edgecolors='white', linewidths=0.5)

        tick_marks = np.arange(n_cls)
        ax.set_xticks(tick_marks + 0.5)
        ax.set_yticks(tick_marks + 0.5)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=fontsize)
        ax.set_yticklabels(class_names, fontsize=fontsize)
        ax.invert_yaxis()

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count' if not normalize else 'Proportion', rotation=270, labelpad=20)

        ax.set_title(title, fontsize=fontsize + 2, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=fontsize)
        ax.set_ylabel('True Label', fontsize=fontsize)

        if show_values:
            thresh = cm_display.max() / 2.0 if not normalize else 0.5
            for i in range(n_cls):
                for j in range(n_cls):
                    val = cm_display[i, j]
                    text = f"{val:{value_format}}"
                    color = 'white' if val > thresh else 'black'
                    ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                            color=color, fontsize=fontsize - 2)

        if n_cls <= 20:
            for i in range(n_cls):
                rect = plt.Rectangle((i, i), 1, 1, fill=False,
                                     edgecolor='gold', linewidth=2, alpha=0.5)
                ax.add_patch(rect)

        fig.tight_layout()

        if save_path:
            save_dir = os.path.dirname(str(save_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"📁 Confusion matrix saved to {save_path}")

        plt.close(fig)
        return fig


# 向后兼容别名（旧代码 import TrainingVisualizer 仍可工作）
TrainingVisualizer = Visualizer

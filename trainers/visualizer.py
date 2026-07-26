"""
训练可视化模块：训练曲线、学习率曲线、混淆矩阵绘制与测试报告打印。

设计原则：
    - 构造器只持有"展示配置"（输出目录、类别名、日志器），一次配置、处处复用；
    - 训练数据（损失历史、混淆矩阵、指标字典）由方法参数显式传入，
      不持有 Trainer 引用，保持 Trainer -> Visualizer 的单向依赖；
    - 因此本类可脱离训练器独立使用（如在 notebook 中复盘实验数据）。

使用示例：
    viz = TrainingVisualizer(save_dir='./output', class_names=['cat', 'dog'])
    viz.plot_acc_loss(train_loss, val_loss, val_acc, val_epochs,
                      save_path='./output/acc_loss.png')
    viz.plot_confusion_matrix(cm, normalize=True, save_path='./output/cm.png')
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def fmt_value(
    val: Optional[float],
    pattern: str = ".4f",
    default: str = "N/A",
    scale: float = 1.0,
    suffix: str = "") -> str:
    """
    安全格式化数值（None / 非法值返回 default）

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


class TrainingVisualizer:
    """
    训练过程可视化器。

    Args:
        save_dir: 默认输出目录（各 plot 方法未显式传 save_path 时使用）
        class_names: 类别名称列表（混淆矩阵与测试报告使用）
        logger: 日志器（None 时使用模块级 logger）
    """

    def __init__(self,
                 save_dir: Optional[Union[str, Path]] = None,
                 class_names: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        self.class_names = class_names or []
        self.logger = logger or logging.getLogger(__name__)

    def _resolve_path(self, save_path, default_name: str) -> Optional[str]:
        """save_path 为 None 且配置了 save_dir 时，落到 save_dir/default_name。"""
        if save_path is not None:
            return str(save_path)
        if self.save_dir is not None:
            return str(self.save_dir / default_name)
        return None

    def plot_acc_loss(self,
                      train_loss: Sequence[float],
                      val_loss: Sequence[float],
                      val_acc: Sequence[float],
                      val_epochs: Optional[Sequence[int]] = None,
                      save_path: Optional[str] = None) -> None:
        """
        绘制训练/验证损失曲线和验证准确率曲线。

        训练阶段不计算精度指标，因此准确率子图仅展示验证集曲线。

        Args:
            train_loss: 每个 epoch 的训练损失
            val_loss: 每次验证的损失
            val_acc: 每次验证的准确率
            val_epochs: 每次验证对应的 epoch（eval_interval > 1 时曲线横轴对齐用；
                        None 时按 1..len(val_loss) 顺序排布）
            save_path: 保存路径（None 时使用 save_dir/acc_loss.png）
        """
        if not (train_loss and val_loss):
            raise ValueError("Loss history is empty, run fit() first.")
        if val_epochs is None:
            val_epochs = list(range(1, len(val_loss) + 1))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_loss) + 1),
                 train_loss, 'ro-', label='Train Loss')
        plt.plot(val_epochs, val_loss, 'bs-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(val_epochs, val_acc, 'bs-', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = self._resolve_path(save_path, 'acc_loss.png')
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_lr_history(self,
                        lr_history: Sequence[float],
                        save_path: Optional[str] = None) -> None:
        """
        绘制学习率变化曲线（对数坐标）。

        Args:
            lr_history: 每个 epoch 的学习率
            save_path: 保存路径（None 时使用 save_dir/lr_curve.png）
        """
        if not lr_history:
            return

        plt.figure(figsize=(8, 4))
        plt.plot(lr_history, 'bo-', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate (log scale)')
        plt.yscale('log')  # 对数坐标更清晰
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = self._resolve_path(save_path, 'lr_curve.png')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"📈 LR curve saved to {save_path}")
        plt.close()

    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              normalize: bool = False,
                              title: str = 'Confusion Matrix',
                              cmap: str = 'Blues',
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
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
            figsize: 画布大小
            fontsize: 字体大小
            show_values: 是否显示数值
            value_format: 数值格式

        Returns:
            matplotlib Figure 对象
        """
        class_names = class_names or self.class_names
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
            save_dir = os.path.dirname(str(save_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"📁 Confusion matrix saved to {save_path}")

        plt.close(fig)
        return fig

    def print_test_report(self,
                          results: Dict[str, Any],
                          confusion_matrix: np.ndarray,
                          elapsed_time: float,
                          speed: float,
                          is_classification: bool = True) -> None:
        """
        打印格式化的测试报告。

        Args:
            results: Metrics.compute() 返回的指标字典
                     （oa/mpa/mf1 及每类 recall_i/precision_i/f1_i，
                       分割任务额外含 miou/fwiou）
            confusion_matrix: 混淆矩阵 (num_classes × num_classes)
            elapsed_time: 测试耗时（秒）
            speed: 测试吞吐（samples/sec）
            is_classification: 分类任务为 True；分割任务额外打印 mIoU/FWIoU
        """
        num_classes = confusion_matrix.shape[0]

        print("\n" + "=" * 60)
        print("🧪 TEST REPORT".center(60))
        print("=" * 60)

        print(f"\n📊 Overall Metrics:")
        print(f"  • Accuracy (OA): {fmt_value(results.get('oa'), '.2f', scale=100, suffix='%')}")
        print(f"  • Mean PA      : {fmt_value(results.get('mpa'), '.2f', scale=100, suffix='%')}")
        print(f"  • Mean F1      : {fmt_value(results.get('mf1'), '.2f', scale=100, suffix='%')}")
        if not is_classification:
            print(f"  • Mean IoU     : {fmt_value(results.get('miou'), '.2f', scale=100, suffix='%')}")
            print(f"  • FW IoU       : {fmt_value(results.get('fwiou'), '.2f', scale=100, suffix='%')}")

        # 每类指标（来自 Metrics.compute 的 recall_i / precision_i / f1_i）
        print(f"\n📋 Per-Class Metrics (Recall / Precision / F1):")
        for cls_idx in range(num_classes):
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Class-{cls_idx}"
            recall = results.get(f'recall_{cls_idx}')
            precision = results.get(f'precision_{cls_idx}')
            f1 = results.get(f'f1_{cls_idx}')
            print(f"  • {cls_name:12s}: "
                  f"{fmt_value(recall, '.2f', scale=100, suffix='%')} / "
                  f"{fmt_value(precision, '.2f', scale=100, suffix='%')} / "
                  f"{fmt_value(f1, '.2f', scale=100, suffix='%')}")

        # 性能指标
        print(f"\n⚡ Performance:")
        print(f"  • Time         : {fmt_value(elapsed_time, '.2f', suffix='s')}")
        print(f"  • Speed        : {fmt_value(speed, '.0f', suffix=' samples/sec')}")

        # 混淆矩阵摘要
        diag = float(np.trace(confusion_matrix))
        total = float(confusion_matrix.sum())
        print(f"\n🔍 Confusion Matrix Summary:")
        print(f"  • Diagonal (correct)   : {int(diag)}")
        print(f"  • Off-diagonal (error) : {int(total - diag)}")

        print("=" * 60 + "\n")

"""
测试报告生成模块：将指标字典与混淆矩阵组装为格式化的文本报告。

设计原则：
    - 纯函数式设计：不持有 Trainer / Visualizer 引用，所有数据由参数显式传入；
    - 可脱离训练器独立使用（如在 notebook 中复盘实验结果、对比多次运行）；
    - 汇总指标从 results 字典动态提取，新增指标只需扩展 _CLS_METRICS / _SEG_METRICS。

使用示例：
    report = print_test_report(
        results={'test/acc': 0.92, 'test/f1': 0.90},
        confusion_matrix=cm,
        elapsed_time=1.5,
        speed=1000,
        class_names=['cat', 'dog'],
        prefix='test/',
        save_path='./output/test_report.txt',
    )
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 格式化工具
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 各任务类型的汇总指标展示顺序与格式
# (key_suffix, display_label, fmt_pattern, scale, suffix)
# ---------------------------------------------------------------------------

_CLS_METRICS: List[Tuple[str, str, str, float, str]] = [
    ('acc',           'Accuracy',      '.2f', 100, '%'),
    ('balanced_acc',  'Balanced Acc',  '.2f', 100, '%'),
    ('f1',            'F1',            '.2f', 100, '%'),
    ('kappa',         'Cohen Kappa',   '.4f', 1,   ''),
]

_SEG_METRICS: List[Tuple[str, str, str, float, str]] = [
    ('oa',   'Accuracy (OA)', '.2f', 100, '%'),
    ('iou',  'IoU',           '.2f', 100, '%'),
    ('f1',   'F1',            '.2f', 100, '%'),
]


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def print_test_report(
    results: Dict[str, Any],
    confusion_matrix: np.ndarray,
    elapsed_time: float,
    speed: float,
    class_names: Optional[List[str]] = None,
    prefix: str = '',
    is_classification: bool = True,
    save_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    生成格式化的测试报告。

    报告先组装为完整字符串，写入独立报告文件（test_report.txt），
    同时在日志中打印一行摘要，避免日志文件被大段报告刷屏。

    汇总指标从 results 字典中动态提取：遍历预定义的指标列表，
    只打印 results 中实际存在的指标，新增指标只需扩展 _CLS_METRICS / _SEG_METRICS。

    Args:
        results: compute() 返回的指标字典（键含 phase 前缀，如 'test/acc'）
        confusion_matrix: 混淆矩阵 (num_classes × num_classes)
        elapsed_time: 测试耗时（秒）
        speed: 测试吞吐（samples/sec）
        class_names: 类别名称列表（逐类指标显示用；None 时显示 Class-0, Class-1, ...）
        prefix: 指标键名前缀（如 'test/'），用于从 results 中查找带前缀的键
        is_classification: 任务类型，决定汇总指标的打印集合
        save_path: 报告文件保存路径；None 时不写文件
        logger: 日志器（None 时使用模块级 logger）

    Returns:
        报告文本（不含日志前缀），供调用方另作他用
    """
    _logger = logger or logging.getLogger(__name__)
    class_names = class_names or []
    num_classes = confusion_matrix.shape[0]
    lines: List[str] = []

    lines.append("=" * 60)
    lines.append("🧪 TEST REPORT".center(60))
    lines.append("=" * 60)

    # ---- 汇总指标（从 results 动态提取，只打印存在的键） ----
    lines.append("")
    lines.append("📊 Overall Metrics:")
    metric_specs = _CLS_METRICS if is_classification else _SEG_METRICS
    summary_parts: List[str] = []  # 收集日志摘要用
    for key_suffix, display_label, fmt_pat, scale, suffix in metric_specs:
        val = results.get(f'{prefix}{key_suffix}')
        if val is not None:
            formatted = fmt_value(val, fmt_pat, scale=scale, suffix=suffix)
            lines.append(f"  • {display_label:14s}: {formatted}")
            summary_parts.append(f"{display_label}: {formatted}")

    # ---- 逐类指标：从混淆矩阵计算 ----
    lines.append("")
    lines.append("📋 Per-Class Metrics (Recall / Precision / F1):")
    eps = 1e-10
    for cls_idx in range(num_classes):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class-{cls_idx}"
        # 从混淆矩阵计算逐类指标
        tp = float(confusion_matrix[cls_idx, cls_idx])
        fn = float(confusion_matrix[cls_idx, :].sum()) - tp
        fp = float(confusion_matrix[:, cls_idx].sum()) - tp
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        lines.append(f"  • {cls_name:12s}: "
                     f"{fmt_value(recall, '.2f', scale=100, suffix='%')} / "
                     f"{fmt_value(precision, '.2f', scale=100, suffix='%')} / "
                     f"{fmt_value(f1, '.2f', scale=100, suffix='%')}")

    # ---- 性能指标 ----
    lines.append("")
    lines.append("⚡ Performance:")
    lines.append(f"  • Time         : {fmt_value(elapsed_time, '.2f', suffix='s')}")
    lines.append(f"  • Speed        : {fmt_value(speed, '.0f', suffix=' samples/sec')}")

    # ---- 混淆矩阵摘要 ----
    diag = float(np.trace(confusion_matrix))
    total = float(confusion_matrix.sum())
    lines.append("")
    lines.append("🔍 Confusion Matrix Summary:")
    lines.append(f"  • Diagonal (correct)   : {int(diag)}")
    lines.append(f"  • Off-diagonal (error) : {int(total - diag)}")

    lines.append("=" * 60)

    report = "\n".join(lines)

    # 写入独立报告文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report + '\n')
        _logger.info(f"📄 Test report saved to {save_path}")

    # train.log 只打印一行摘要，避免大段报告刷屏
    summary = " | ".join(summary_parts) if summary_parts else "N/A"
    _logger.info(f"🧪 Test completed | {summary}")

    return report

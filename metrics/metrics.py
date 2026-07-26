"""
通用分割/分类任务的评价指标模块。

核心思路：
    所有指标均基于 **混淆矩阵 (Confusion Matrix)** 推导。
    混淆矩阵 shape 为 (num_classes, num_classes)，
    其中 cm[i, j] 表示真实类别为 i、被预测为 j 的样本数量。

使用流程：
    metric = SegmentationMetric(num_classes=21)
    for preds, targets in dataloader:
        metric.update(preds, targets)
    results = metric.compute()
    metric.reset()
"""

from typing import Optional, Dict
import torch


class Metrics:
    """
    基于混淆矩阵的通用分类/分割指标计算器。

    支持指标：
        - OA  (Overall Accuracy)    总体精度
        - mPA (Mean Pixel Accuracy) 平均类别精度（即 mean Recall）
        - mIoU (Mean IoU)           平均交并比
        - FWIoU (Freq Weighted IoU) 频率加权交并比
        - Precision / Recall / F1   各类精确率、召回率、F1

    Args:
        num_classes: 类别总数
        ignore_index: 忽略的标签索引（如 255 表示未标注区域），不参与计算
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 混淆矩阵，使用 float64 避免大样本量时精度丢失
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.float64
        )

    def reset(self):
        """重置混淆矩阵，开始新一轮评估。"""
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        累积一个 batch 的预测结果到混淆矩阵。

        Args:
            preds: 预测结果。
                   - 若为 logits/probabilities (N, C, ...) 则自动取 argmax
                   - 若为类别索引 (N, ...) 则直接使用
            targets: 真实标签，shape 为 (N, ...) 的整数张量
        """
        # 如果 preds 比 targets 多一个维度，说明是 logits，取 argmax
        if preds.dim() == targets.dim() + 1:
            preds = preds.argmax(dim=1)

        preds = preds.flatten().long()
        targets = targets.flatten().long()

        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            preds = preds[valid]
            targets = targets[valid]

        # 构建混淆矩阵：行=真实类别，列=预测类别
        # 利用 bincount 高效统计
        indices = targets * self.num_classes + preds
        cm_flat = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm_flat.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """
        根据累积的混淆矩阵计算所有指标。

        Returns:
            字典，包含 oa, mpa, miou, fwiou 及各指标
        """
        cm = self.confusion_matrix
        eps = 1e-10  # 防止除零

        # 对角线 = TP（每类的正确预测数）
        tp = torch.diag(cm)
        # 每行求和 = 该类的真实样本总数（GT count）
        gt_count = cm.sum(dim=1)
        # 每列求和 = 被预测为该类的样本总数（Pred count）
        pred_count = cm.sum(dim=0)

        # ---- Overall Accuracy (OA) ----
        oa = tp.sum() / (cm.sum() + eps)

        # ---- Per-class Recall (Pixel Accuracy per class) ----
        recall = tp / (gt_count + eps)
        mpa = recall.mean()

        # ---- Per-class Precision ----
        precision = tp / (pred_count + eps)

        # ---- Per-class IoU ----
        # IoU = TP / (GT + Pred - TP)
        iou = tp / (gt_count + pred_count - tp + eps)
        miou = iou.mean()

        # ---- Frequency Weighted IoU ----
        freq = gt_count / (cm.sum() + eps)
        fwiou = (freq * iou).sum()

        # ---- Per-class F1 ----
        f1 = 2 * precision * recall / (precision + recall + eps)
        mf1 = f1.mean()

        results = {
            'oa': oa.item(),
            'mpa': mpa.item(),
            'miou': miou.item(),
            'fwiou': fwiou.item(),
            'mf1': mf1.item(),
        }

        # 附加各类别详细指标
        for i in range(self.num_classes):
            results[f'iou_{i}'] = iou[i].item()
            results[f'precision_{i}'] = precision[i].item()
            results[f'recall_{i}'] = recall[i].item()
            results[f'f1_{i}'] = f1[i].item()

        return results


class SegmentationMetric(Metrics):
    """
    语义分割专用指标（与 Metrics 完全一致，语义别名）。

    语义分割中每个像素视为一个独立样本，
    输入 preds/targets 的 shape 通常为 (N, H, W)。

    Example:
        >>> metric = SegmentationMetric(num_classes=21, ignore_index=255)
        >>> # preds: (N, 21, H, W) logits 或 (N, H, W) 类别索引
        >>> metric.update(preds, targets)
        >>> results = metric.compute()
        >>> print(f"mIoU: {results['miou']:.4f}")
    """
    pass


class TorchMetricsWrapper:
    """
    torchmetrics 库的轻量包装器（可选依赖）。

    当项目中已安装 torchmetrics 时，可用此类快速接入其丰富的指标实现。
    若未安装则抛出友好提示。

    Example:
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> wrapper = TorchMetricsWrapper(MulticlassJaccardIndex(num_classes=21))
        >>> wrapper.update(preds, targets)
        >>> print(wrapper.compute())
    """

    def __init__(self, metric):
        """
        Args:
            metric: torchmetrics.Metric 实例
        """
        try:
            from torchmetrics import Metric
            assert isinstance(metric, Metric), (
                f"期望 torchmetrics.Metric 实例，实际得到 {type(metric)}"
            )
        except ImportError:
            raise ImportError(
                "TorchMetricsWrapper 需要安装 torchmetrics: pip install torchmetrics"
            )
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute()

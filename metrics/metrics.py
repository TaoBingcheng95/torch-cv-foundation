"""
通用分割/分类任务的评价指标模块。

分层设计：
    ConfusionMatrix（状态层）
        只负责混淆矩阵的累积（update）、重置（reset）、合并（__add__ / all_reduce），
        并以 property 暴露逐类 one-vs-rest 计数视图（tp/fp/fn/tn 向量），
        不推导任何指标。
    指标计算层（后续的 ClassificationMetric / SegmentationMetric）
        持有 ConfusionMatrix，从计数视图推导各自语义正确的指标。

混淆矩阵约定：
    shape 为 (num_classes, num_classes)，
    cm[i, j] 表示真实类别为 i、被预测为 j 的样本数量。
"""
from typing import Optional, Dict
import math
import torch
import torch.distributed as dist



class ConfusionMatrix:
    """
    混淆矩阵状态容器（参数接收层）。

    只做三件事：累积、重置、合并，指标推导交给上层计算类。
    二分类是 num_classes=2 的特例，无需单独的标量计数器。

    Example:
        >>> cm = ConfusionMatrix(num_classes=10)
        >>> for preds, targets in dataloader:
        ...     cm.update(preds, targets)   # logits 或类别索引均可
        >>> cm.tp, cm.fp, cm.fn, cm.tn     # 逐类 one-vs-rest 计数向量

    Args:
        num_classes: 类别总数（>= 2）
        ignore_index: 忽略的标签索引（如分割任务中 255 表示未标注区域），
            None 表示不忽略。分类任务通常无需设置。
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 计数使用 int64：整数统计精确无舍入，且实际样本量远不会溢出
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # ------------------------------------------------------------------
    # 状态维护：累积 / 重置 / 合并
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清零混淆矩阵，开始新一轮统计。"""
        self.matrix.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        累积一个 batch 的预测结果。

        Args:
            preds: 预测结果。
                   - logits/probabilities (N, C, ...) 自动沿 dim=1 取 argmax
                   - 类别索引 (N, ...) 直接使用
            targets: 真实标签，与 argmax 后的 preds 同 shape 的整数张量
        """
        # preds 比 targets 多一维 → 视为 logits，取 argmax
        if preds.dim() == targets.dim() + 1:
            preds = preds.argmax(dim=1)
        if preds.shape != targets.shape:
            raise ValueError(
                f"preds 与 targets 形状不匹配: {tuple(preds.shape)} vs {tuple(targets.shape)}"
            )

        # 统一搬运到矩阵所在设备（CPU），避免与训练设备耦合
        preds = preds.reshape(-1).long().to(self.matrix.device)
        targets = targets.reshape(-1).long().to(self.matrix.device)

        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            preds, targets = preds[valid], targets[valid]

        # 越界标签会破坏 bincount 的索引编码，必须显式报错而非静默丢弃
        for name, t in (("preds", preds), ("targets", targets)):
            if t.numel() and (t.min() < 0 or t.max() >= self.num_classes):
                raise ValueError(
                    f"{name} 存在越界类别索引: 范围 [{t.min()}, {t.max()}]，"
                    f"合法区间 [0, {self.num_classes - 1}]"
                )

        # 行=真实类别，列=预测类别；bincount 批量统计
        indices = targets * self.num_classes + preds
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def _check_compatible(self, other: "ConfusionMatrix") -> None:
        if self.num_classes != other.num_classes:
            raise ValueError(
                f"num_classes 不一致，无法合并: {self.num_classes} vs {other.num_classes}"
            )

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        """合并两个混淆矩阵（如多进程各自统计后离线汇总）。"""
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented
        self._check_compatible(other)
        merged = ConfusionMatrix(self.num_classes, ignore_index=self.ignore_index)
        merged.matrix = self.matrix + other.matrix
        return merged

    def __iadd__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        """就地合并。"""
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented
        self._check_compatible(other)
        self.matrix += other.matrix
        return self

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：对各 rank 的矩阵求和并同步到所有进程。
        未初始化分布式环境时为 no-op，可无条件调用。
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        mat = self.matrix
        # NCCL 后端只支持 GPU 张量
        if dist.get_backend() == dist.Backend.NCCL:
            mat = mat.cuda()
        dist.all_reduce(mat, op=dist.ReduceOp.SUM)
        if mat is not self.matrix:
            self.matrix.copy_(mat.cpu())

    # ------------------------------------------------------------------
    # 派生视图：逐类 one-vs-rest 计数（均为 shape=(num_classes,) 的 int64 向量）
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        """有效样本总数（已排除 ignore_index）。"""
        return int(self.matrix.sum())

    @property
    def gt_count(self) -> torch.Tensor:
        """每类真实样本数（行和）。"""
        return self.matrix.sum(dim=1)

    @property
    def pred_count(self) -> torch.Tensor:
        """每类被预测样本数（列和）。"""
        return self.matrix.sum(dim=0)

    @property
    def tp(self) -> torch.Tensor:
        """每类正确预测数（对角线）。"""
        return self.matrix.diagonal()

    @property
    def fp(self) -> torch.Tensor:
        """每类误报数：被预测为该类但真实为其他类。"""
        return self.pred_count - self.tp

    @property
    def fn(self) -> torch.Tensor:
        """每类漏检数：真实为该类但被预测为其他类。"""
        return self.gt_count - self.tp

    @property
    def tn(self) -> torch.Tensor:
        """每类真阴数：one-vs-rest 视角下的其余部分。"""
        return self.total - self.tp - self.fp - self.fn

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.total})"
        )



class ClassificationMetric:
    """
    分类任务指标计算层。

    持有 ConfusionMatrix 作为唯一累积状态，所有指标从其计数视图推导；
    例外是 Top-k 准确率——它依赖 logits 排序信息，无法从混淆矩阵还原，
    因此单独维护命中/总数两个标量计数。

    支持指标（compute 返回键名）：
        - acc             总体准确率（单标签多分类下恒等于 micro P/R/F1）
        - balanced_acc    平衡准确率（= macro Recall），类别不均衡时更可靠
        - macro_precision / macro_recall / macro_f1   宏平均（各类等权）
        - weighted_f1     按类别真实频率加权的 F1
        - kappa           Cohen's Kappa，扣除随机一致性后的一致度
        - top{k}_acc      Top-k 准确率（仅当 top_k 非空且 update 传入 logits 时）
        - precision_i / recall_i / f1_i               逐类指标

    注意：
        macro 平均对验证集中未出现的类别计为 0（与 sklearn 默认行为一致）。

    Example:
        >>> metric = ClassificationMetric(num_classes=10, top_k=5)
        >>> for logits, targets in dataloader:
        ...     metric.update(logits, targets)
        >>> results = metric.compute()
        >>> print(f"acc={results['acc']:.4f}, top5={results['top5_acc']:.4f}")

    Args:
        num_classes: 类别总数（>= 2）
        top_k: 额外统计 Top-k 准确率，None 表示不统计
        ignore_index: 忽略的标签索引，分类任务通常保持 None
    """

    def __init__(
        self,
        num_classes: int,
        top_k: Optional[int] = None,
        ignore_index: Optional[int] = None,
    ):
        if top_k is not None and not 1 <= top_k <= num_classes:
            raise ValueError(
                f"top_k 需在 [1, num_classes={num_classes}] 内，实际得到 {top_k}"
            )
        self.cm = ConfusionMatrix(num_classes, ignore_index=ignore_index)
        self.top_k = top_k
        self._topk_correct = 0
        self._topk_total = 0

    @property
    def num_classes(self) -> int:
        return self.cm.num_classes

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.matrix

    def reset(self) -> None:
        """重置全部累积状态（混淆矩阵 + Top-k 计数）。"""
        self.cm.reset()
        self._topk_correct = 0
        self._topk_total = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        累积一个 batch。输入约定与 ConfusionMatrix.update 一致；
        若启用了 top_k，仅当 preds 为 logits 形式时才能累积 Top-k 统计。
        """
        # 先累积 Top-k（需在 argmax 前拿到完整 logits），再交给混淆矩阵
        if self.top_k is not None and preds.dim() == targets.dim() + 1:
            self._update_topk(preds, targets)
        self.cm.update(preds, targets)

    def _update_topk(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        # (N, C, ...) 沿类别维取 top-k 索引 → (N, k, ...)
        topk_idx = logits.topk(self.top_k, dim=1).indices
        hit = (topk_idx == targets.unsqueeze(1).long()).any(dim=1)  # (N, ...)
        if self.cm.ignore_index is not None:
            valid = targets != self.cm.ignore_index
            hit = hit[valid]
        self._topk_correct += int(hit.sum())
        self._topk_total += int(hit.numel())

    def all_reduce(self) -> None:
        """DDP 汇总：混淆矩阵 + Top-k 计数一并同步，未初始化时 no-op。"""
        self.cm.all_reduce()
        if not (dist.is_available() and dist.is_initialized()):
            return
        if self.top_k is not None:
            t = torch.tensor([self._topk_correct, self._topk_total], dtype=torch.int64)
            if dist.get_backend() == dist.Backend.NCCL:
                t = t.cuda()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self._topk_correct, self._topk_total = int(t[0]), int(t[1])

    def compute(self) -> Dict[str, float]:
        """
        从累积状态计算全部分类指标。

        Returns:
            汇总指标 + 逐类指标（precision_i/recall_i/f1_i，
            键名格式与 trainer 日志分组正则 `^(.+)_(\\d+)$` 兼容）
        """
        eps = 1e-10  # 防除零：未出现的类别对应指标记为 0
        tp = self.cm.tp.double()
        gt_count = self.cm.gt_count.double()
        pred_count = self.cm.pred_count.double()
        total = self.cm.total

        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        acc = tp.sum() / (total + eps)
        # 按真实频率加权的 F1
        freq = gt_count / (total + eps)
        weighted_f1 = (freq * f1).sum()
        # Cohen's Kappa：pe 为边缘分布下的期望一致率
        pe = (gt_count * pred_count).sum() / (total ** 2 + eps)
        kappa = (acc - pe) / (1 - pe + eps)

        results = {
            'acc': acc.item(),
            'balanced_acc': recall.mean().item(),
            'macro_precision': precision.mean().item(),
            'macro_recall': recall.mean().item(),
            'macro_f1': f1.mean().item(),
            'weighted_f1': weighted_f1.item(),
            'kappa': kappa.item(),
        }
        if self.top_k is not None and self._topk_total > 0:
            results[f'top{self.top_k}_acc'] = self._topk_correct / self._topk_total

        # 逐类详细指标
        for i in range(self.num_classes):
            results[f'precision_{i}'] = precision[i].item()
            results[f'recall_{i}'] = recall[i].item()
            results[f'f1_{i}'] = f1[i].item()

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"top_k={self.top_k}, total={self.cm.total})"
        )



class SegmentationMetric:
    """
    语义分割任务指标计算层。

    与 ClassificationMetric 同构：持有 ConfusionMatrix 作为唯一累积状态，
    所有指标从其计数视图推导。每个像素视为一个独立样本，
    输入 preds/targets 的 shape 通常为 (N, C, H, W) logits 或 (N, H, W) 类别索引。

    支持指标（compute 返回键名）：
        - oa              Overall Accuracy，总体像素精度
        - mpa             Mean Pixel Accuracy，平均类别像素精度（= macro Recall）
        - miou            Mean IoU，平均交并比（分割主监控指标）
        - fwiou           Frequency Weighted IoU，按类别频率加权的 IoU
        - mf1             Mean F1（= mean Dice）
        - iou_i / precision_i / recall_i / f1_i       逐类指标

    Example:
        >>> metric = SegmentationMetric(num_classes=21, ignore_index=255)
        >>> # preds: (N, 21, H, W) logits 或 (N, H, W) 类别索引
        >>> metric.update(preds, targets)
        >>> results = metric.compute()
        >>> print(f"mIoU: {results['miou']:.4f}")

    Args:
        num_classes: 类别总数（含背景类，>= 2）
        ignore_index: 忽略的标签索引，默认 255（VOC 等数据集的 void 区域约定）
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255):
        self.cm = ConfusionMatrix(num_classes, ignore_index=ignore_index)

    @property
    def num_classes(self) -> int:
        return self.cm.num_classes

    @property
    def ignore_index(self) -> Optional[int]:
        return self.cm.ignore_index

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.matrix

    def reset(self) -> None:
        """重置混淆矩阵，开始新一轮评估。"""
        self.cm.reset()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """累积一个 batch，输入约定与 ConfusionMatrix.update 一致。"""
        self.cm.update(preds, targets)

    def all_reduce(self) -> None:
        """DDP 多进程汇总，未初始化时 no-op。"""
        self.cm.all_reduce()

    def compute(self) -> Dict[str, float]:
        """
        从累积状态计算全部分割指标。

        Returns:
            汇总指标（oa/mpa/miou/fwiou/mf1）+ 逐类指标
            （iou_i/precision_i/recall_i/f1_i，键名格式与 trainer
            日志分组正则 `^(.+)_(\\d+)$` 兼容）
        """
        eps = 1e-10  # 防除零：未出现的类别对应指标记为 0
        tp = self.cm.tp.double()
        gt_count = self.cm.gt_count.double()
        pred_count = self.cm.pred_count.double()
        total = self.cm.total

        # ---- Overall Accuracy ----
        oa = tp.sum() / (total + eps)

        # ---- 逐类 Recall（Pixel Accuracy per class）与 Precision ----
        recall = tp / (gt_count + eps)
        precision = tp / (pred_count + eps)
        mpa = recall.mean()

        # ---- 逐类 IoU: TP / (GT + Pred - TP) ----
        iou = tp / (gt_count + pred_count - tp + eps)
        miou = iou.mean()

        # ---- Frequency Weighted IoU ----
        freq = gt_count / (total + eps)
        fwiou = (freq * iou).sum()

        # ---- 逐类 F1（= Dice）----
        f1 = 2 * precision * recall / (precision + recall + eps)
        mf1 = f1.mean()

        results = {
            'oa': oa.item(),
            'mpa': mpa.item(),
            'miou': miou.item(),
            'fwiou': fwiou.item(),
            'mf1': mf1.item(),
        }

        # 逐类详细指标
        for i in range(self.num_classes):
            results[f'iou_{i}'] = iou[i].item()
            results[f'precision_{i}'] = precision[i].item()
            results[f'recall_{i}'] = recall[i].item()
            results[f'f1_{i}'] = f1[i].item()

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.cm.total})"
        )



def _kappa_from_matrix(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """
    从混淆矩阵计算 Cohen's Kappa。

    边界约定与 SCD 官方实现一致：矩阵全零或期望一致率 pe 为 1 时返回 0。

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别
    """
    hist = matrix.double()
    total = hist.sum()
    if total == 0:
        return 0.0
    po = hist.diagonal().sum() / total
    pe = (hist.sum(dim=1) * hist.sum(dim=0)).sum() / total ** 2
    if abs(1 - pe) < eps:
        return 0.0
    return ((po - pe) / (1 - pe)).item()



def separated_kappa(matrix: torch.Tensor, bg_index: int = 0) -> Dict[str, float]:
    """
    分离 Kappa（Separated Kappa, SeK）系数。

    面向"多类前景 + 主导性背景"的场景（语义变化检测 SCD、灾损分级等）：
    传统 Kappa 会被巨量的"背景→背景"正确项（TN）虚高，SeK 将其从混淆
    矩阵中剔除后再计算 Kappa（衡量前景类语义分对了没），并乘以前景二值
    IoU 的指数惩罚项（衡量前景空间定位准不准）：

        SeK = kappa_n0 * exp(IoU_fg - 1)

    参考：SECOND 数据集官方指标 https://captain-whu.github.io/SCD/

    注意：
        - 前景仅 1 类时 kappa_n0 退化为 IoU 的变体，此时直接用 IoU/F1 即可；
        - 类别相对均衡的普通多分类任务不适用，应使用传统 Kappa。

    Example:
        >>> cm = ConfusionMatrix(num_classes=5)
        >>> cm.update(preds, targets)
        >>> results = separated_kappa(cm.matrix)
        >>> print(f"SeK: {results['sek']:.5f}")

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别，
            可直接传入 ConfusionMatrix.matrix 或 SegmentationMetric.confusion_matrix
        bg_index: 背景/未变化类的索引，默认 0（SCD 官方约定）

    Returns:
        {
            'sek':      分离 Kappa 系数,
            'kappa_n0': 剔除背景 TN 后的 Kappa（前景语义分类一致度）,
            'iou_fg':   前景（变化区域）二值 IoU,
            'iou_bg':   背景（未变化区域）二值 IoU,
            'biou':     二值 mIoU = (iou_fg + iou_bg) / 2,
        }
    """
    hist = matrix.double()
    n = hist.shape[0]
    if hist.dim() != 2 or hist.shape[0] != hist.shape[1]:
        raise ValueError(f"matrix 需为方阵，实际 shape={tuple(matrix.shape)}")
    if not 0 <= bg_index < n:
        raise ValueError(f"bg_index 越界: {bg_index}，合法区间 [0, {n - 1}]")

    eps = 1e-10
    # 剔除"背景→背景"的巨量 TN，衡量前景类语义分类一致度
    hist_n0 = hist.clone()
    hist_n0[bg_index, bg_index] = 0
    kappa_n0 = _kappa_from_matrix(hist_n0)

    # 折叠为"背景/前景"二值矩阵，计算前景空间定位质量（行=真实，列=预测）
    tn = hist[bg_index, bg_index]                # 背景判对
    fn = hist[:, bg_index].sum() - tn            # 前景漏检：真前景 → 预测背景
    fp = hist[bg_index, :].sum() - tn            # 背景误检：真背景 → 预测前景
    tp = hist.sum() - tn - fp - fn               # 前景判为前景（含类间混淆）

    iou_fg = (tp / (tp + fp + fn + eps)).item()
    iou_bg = (tn / (tn + fp + fn + eps)).item()
    sek = kappa_n0 * math.exp(iou_fg - 1)

    return {
        'sek': sek,
        'kappa_n0': kappa_n0,
        'iou_fg': iou_fg,
        'iou_bg': iou_bg,
        'biou': (iou_fg + iou_bg) / 2,
    }



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

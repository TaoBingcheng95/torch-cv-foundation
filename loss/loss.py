
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits.
    """

    def __init__(
        self,
        reduction: str ="mean",
        weight: Optional[float]=None,
        pos_weight: Optional[float]=None):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        if reduction not in ("none", "mean", "sum"):
            raise ValueError()
        self.reduction = reduction
        if weight is not None:
            self.register_buffer(
                'weight_tensor',
                torch.tensor(weight, dtype=torch.float32)
            )
        else:
            self.weight_tensor = None
        # pos_weight 也注册为 buffer，避免每次 forward 重复创建张量
        if pos_weight is not None:
            self.register_buffer(
                'pos_weight_tensor',
                torch.tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.pos_weight_tensor = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=self.weight_tensor,
            reduction="none",
            pos_weight=self.pos_weight_tensor,
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
                        


class CEWithLogitsLoss(nn.Module):
    """
    Cross Entropy Loss (接受原始 logits，内部执行 Softmax).
    适用于多分类互斥任务（如语义分割、图像分类）。
    """
    def __init__(self, weight=None, 
                 reduction='mean', 
                 ignore_index=255):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, 
                                                 reduction=reduction,
                                                 ignore_index=ignore_index)


    def forward(self, output, target):
        loss = self.cross_entropy(output, target)
        return loss

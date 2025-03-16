import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn


def make_one_hot(labels, classes):
    one_hot = torch.zeros(size=(labels.size()[0], classes, labels.size()[2], labels.size()[3]), device=labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def dice_coeff(inputs: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert inputs.size() == target.size()
    if inputs.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {inputs.shape})')

    if inputs.dim() == 2 or reduce_batch_first:
        inter = torch.dot(inputs.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(inputs) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(inputs.shape[0]):
            dice += dice_coeff(inputs[i, ...], target[i, ...])
        return dice / inputs.shape[0]


def multiclass_dice_coeff(inputs: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert inputs.size() == target.size()
    dice = 0
    for channel in range(inputs.shape[1]):
        dice += dice_coeff(inputs[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
    return dice / inputs.shape[1]


def dice_loss(inputs: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert inputs.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(inputs, target, reduce_batch_first=True)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


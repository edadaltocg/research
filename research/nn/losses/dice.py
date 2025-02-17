from typing import Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F

reduction_ops = {"mean": torch.mean, "sum": torch.sum, "none": lambda x: x}


def dice_coef_loss(input, target, num_classes=2, dims=(1, 2), smooth=1e-8):
    """Smooth Dice coefficient + Cross-entropy loss function."""

    ground_truth_oh = F.one_hot(target, num_classes=num_classes)
    prediction_norm = F.softmax(input, dim=1).permute(0, 2, 3, 1)

    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)

    dice = 2.0 * intersection / (summation + smooth)
    dice_mean = dice.mean(dim=-1)
    dice_loss = torch.mean(1 - dice_mean)

    ce_loss = F.cross_entropy(input, target)

    return dice_loss + ce_loss


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-8,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.eps = eps

        self._reduction_op = reduction_ops[reduction]

    def forward(self, inputs: Tensor, targets: Tensor):
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.view(inputs.shape[0], self.num_classes, -1)

        # `F.one_hot` appends the class dimension at the end.
        targets = F.one_hot(targets, num_classes=self.num_classes)
        targets = targets.permute(0, -1, 1, *list(range(targets.ndim))[2:-1]).float()
        targets = targets.view(targets.shape[0], self.num_classes, -1)

        intersection = (inputs * targets).sum(dim=-1)
        union = inputs.sum(dim=-1) + targets.sum(dim=-1)

        dsc = 2 * intersection / (union + self.eps)
        loss = 1 - dsc.mean(dim=1)
        return self._reduction_op(loss)

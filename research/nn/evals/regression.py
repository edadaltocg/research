import torch
from torch import Tensor


def residual(pred: Tensor, target: Tensor):
    return pred - target


def r_squared_math(pred: Tensor, target: Tensor):
    """Usually between 0 and 1.

    A naive regressor that outputs always the mean target will have an R^2 of zero.
    """
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2)
    return 1 - ss_res / ss_tot

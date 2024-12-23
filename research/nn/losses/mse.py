import torch
from torch import Tensor


def se(logits: Tensor, targets: Tensor):
    return (logits - targets) ** 2


def mse_loss(logits: Tensor, targets: Tensor):
    r"""
    $$
    f(X,Y) = 1/N \sum_i^N{ ( X_i - Y_i ) ^ 2 }
    $$
    """

    return se(logits, targets).mean()


def mat_se(logits: Tensor, targets: Tensor):
    return (
        torch.einsum("ij,jk->ik", logits.T, logits)
        - 2 * torch.einsum("ij,jk->ik", logits.T, targets)
        + torch.einsum("ij,jk->ik", targets.T, targets)
    )


def matrix_mse_loss(logits: Tensor, targets: Tensor):
    return torch.squeeze(mat_se(logits, targets) / logits.size(0))

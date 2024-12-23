"""Generative models: Gaussian Discriminant Analysis."""

import torch
from torch import Tensor, nn


def covariance_math(x: Tensor):
    r"""
    $$
    Cov(X) = \E(XX^\top) - \E[X]\E[X]^\top
    $$

    Args:
        x: `Tensor[n,d]`
    """
    mean_x = torch.mean(x, dim=0, keepdim=True)

    # \E(XX^\top)
    outer_x = torch.einsum("ik,jk->ij", x, x) / x.size(0)

    # E[X] * E[X]^\top
    outer_mean_x = torch.einsum("ik,jk->ij", mean_x, mean_x)

    # Cov(X) = \E(XX^\top) - \E[X]\E[X]^\top
    return outer_x - outer_mean_x


def mean_math(x: Tensor):
    return torch.einsum("ij->j", x).squeeze(-1) / x.size(0)


class GaussianDiscriminantAnalysis(nn.Module):
    def __init__(self, dim: int, nclasses: int) -> None:
        super().__init__()

        self.phi = nn.Parameter(torch.randn(1), requires_grad=True)
        self.mu_0 = nn.Parameter(torch.randn(dim, 1), requires_grad=True)
        self.mu_1 = nn.Parameter(torch.randn(dim, 1), requires_grad=True)
        self.sigma = nn.Parameter(torch.diag(torch.ones(dim)), requires_grad=True)

    def forward(self, x: Tensor):
        return

    @classmethod
    def mle(self, x: Tensor, y: Tensor):
        return

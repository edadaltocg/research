"""Generalized linear models."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LinearRegression(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.n = dim
        self.theta = nn.Parameter(torch.randn(dim, 1), requires_grad=True)

    def forward(self, x: Tensor):
        """
        Args:
            `x: Tensor[n,d]`

        Returns:
            Tensor[n,1]
        """
        return torch.einsum("ij,jk->ik", x, self.theta)

    @classmethod
    def analytical_solution(cls, x: Tensor, y: Tensor):
        n = x.size(1)
        model = cls(n)
        xT = x.T
        xTy = torch.mm(xT, y)
        xTx = torch.mm(xT, x)
        model.theta.data = torch.mm(torch.inverse(xTx), xTy)
        return model


class LogisticRegression(nn.Module):
    r"""
    Probabilistic interpretation:

        $$
        y|x;\theta \sim Bernoulli(\phi) = \phi^y (1-\phi)^(1-y)
        $$
    """

    def __init__(self, dim):
        super().__init__()

        self.n = dim
        self.theta = nn.Parameter(torch.randn(dim, 1), requires_grad=True)

    def forward(self, x: Tensor):
        """
        Args:
            `x: Tensor[n,d]`

        Returns:
            Tensor[n,1]
        """
        return torch.sigmoid(torch.einsum("ij,jk->ik", x, self.theta))


class SoftmaxRegression(nn.Module):
    r"""
    Probabilistic Interpretation:

        $$
        y|x;\theta \sim Categorical(\phi_1, \dots, \phi_{k-1}) = \phi_1^{\1{x=1}} \times \dots \times \phi_k^{\1{x=k}}
        $$
    """

    def __init__(self, dim):
        super().__init__()

        self.n = dim
        self.theta = nn.Parameter(torch.randn(dim, 1), requires_grad=True)

    def forward(self, x: Tensor):
        """
        Args:
            `x: Tensor[n,d]`

        Returns:
            Tensor[n,1]
        """
        return F.softmax(torch.einsum("ij,jk->ik", x, self.theta))

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ReLUMath(nn.Module):
    r"""
    Rectified linear unit.

    $$
    f(x) = max\{x,0\}
    $$
    """

    def forward(self, x: Tensor):
        return torch.max(x, torch.zeros_like(x))


class GELUMath(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)

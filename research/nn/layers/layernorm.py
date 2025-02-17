import torch
from torch import Tensor, nn

nn.LayerNorm()


class LayerNorm(nn.Module):
    """
    $$
    y = \frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \beta
    $$
    """

    def __init__(
        self,
        normalized_shape: int | tuple,
        eps=1e-05,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        return self.weight * (x - mean) / std + self.bias

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output

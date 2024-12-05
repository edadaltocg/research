from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        inner_dim: int,
        dropout_p=0.1,
        activation: Callable = F.relu,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.activation = activation
        self.dropout_p = dropout_p

        self.linear1 = nn.Linear(embed_dim, inner_dim, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(inner_dim, embed_dim, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout_p)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class LinearClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class LLaMAMLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x_fc_1 = self.linear1(x)
        x_fc_2 = self.linear2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)

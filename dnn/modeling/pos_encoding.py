import torch
from torch import Tensor, nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.empty(1, seq_len, hidden_dim).normal_(std=0.02))

    def forward(self, x: Tensor) -> Tensor:
        return self.pe


class VanillaPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        pe = torch.zeros(1, max_len, hidden_dim)
        pos = torch.arange(max_len).float().view(-1, 1)
        dim = torch.arange(hidden_dim // 2).float().view(1, -1)
        # 2t
        pe[:, :, 0::2] = torch.sin(pos / (10_000 ** (2 * dim / hidden_dim)))
        # 2t + 1
        pe[:, :, 1::2] = torch.cos(pos / (10_000 ** (2 * dim / hidden_dim)))

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.size()
        return self.pe
        # return self.pe[:, :s, :] non traceable


class EmbeddingPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(x.size(1), device=x.device).long()
        return self.pe(pos)


def vanilla_positional_encoding(x):
    _, seq_len, d_model = x.size()
    pe = torch.zeros(1, seq_len, d_model)
    pos = torch.arange(seq_len).float().view(-1, 1)
    dim = torch.arange(d_model // 2).float().view(1, -1)
    # 2t
    pe[:, :, 0::2] = torch.sin(pos / (10_000 ** (2 * dim / d_model)))
    # 2t + 1
    pe[:, :, 1::2] = torch.cos(pos / (10_000 ** (2 * dim / d_model)))
    return pe


def positional_encoding_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
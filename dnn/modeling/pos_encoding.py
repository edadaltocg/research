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


# RoPE
def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10_000) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(q: Tensor, k: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)


class RotaryPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.freqs_cis = precompute_freqs_cis(hidden_dim, max_len)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return apply_rotary_emb(q, k, self.freqs_cis)
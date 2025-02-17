import path
from typing import Callable
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout_p: float = 0,
        bias: bool = True,
        sdpa: Callable = F.scaled_dot_product_attention,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_p = dropout_p
        self.bias = bias
        self.sdpa = sdpa

        self.head_dim = embed_dim // num_q_heads
        self.scale = 1 / math.sqrt(embed_dim)
        self.kv_repeats = num_q_heads // num_kv_heads

        self.wq = nn.Linear(
            embed_dim,
            num_q_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.wk = nn.Linear(
            embed_dim,
            num_kv_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.wv = nn.Linear(
            embed_dim,
            num_kv_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)

        k = torch.repeat_interleave(k, self.repeats, dim=2)
        v = torch.repeat_interleave(v, self.repeats, dim=2)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        x = self.sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


GQA = GroupedQueryAttention

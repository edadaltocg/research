import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0,
        bias: bool = True,
        kv_cache: None = None,
        sdpa: Callable = F.scaled_dot_product_attention,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.bias = bias
        self.kv_cache = kv_cache
        self.sdpa = sdpa
        # if self.is_causal:
        #     self.register_buffer("attn_mask", torch.tril(torch.ones(1024, 1024))

        self.qkv_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.qkv_dim)

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        batch_size, _, _ = x.size()

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, -1, self.num_heads, self.qkv_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.qkv_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.qkv_dim).transpose(1, 2)

        if self.kv_cache is not None and input_pos is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

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
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        x = self.out_proj(x)
        x = self.dropout(x)  # necessary?
        return x


MultiheadAttention = MHA

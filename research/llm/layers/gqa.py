from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from research.nn.layers.gqa import GroupedQueryAttention


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(3))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class KVCache(nn.Module):
    k: Tensor
    v: Tensor

    def __init__(
        self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, dtype=torch.float32, device=None
    ):
        super().__init__()
        self.register_buffer(
            "k", torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device)
        )
        self.register_buffer(
            "v", torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device)
        )

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor):
        self.k[:, :, input_pos] = k_val
        self.v[:, :, input_pos] = v_val
        return self.k[:, :, : input_pos[-1] + 1], self.v[:, :, : input_pos[-1] + 1]


class GroupedQueryAttentionWithRoPE(GroupedQueryAttention):
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        freqs_cis: Tensor = args[0] if len(args) > 0 else kwargs["freqs_cis"]
        attn_mask: Tensor | None = args[1] if len(args) > 1 else kwargs.get("attn_mask")

        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        dropout_p = self.dropout_p if self.training else 0

        x = self.sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


class GroupedQueryAttentionWithRoPEAndCache(GroupedQueryAttentionWithRoPE):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        kv_cache: KVCache | None = None,
        dropout_p: float = 0,
        bias: bool = False,
        sdpa: Callable = F.scaled_dot_product_attention,
        device=None,
        dtype=None,
    ):
        super().__init__(
            embed_dim,
            num_q_heads,
            num_kv_heads,
            dropout_p,
            bias,
            sdpa,
            device,
            dtype,
        )

        self.kv_cache = kv_cache
        head_dim = embed_dim // num_q_heads
        self.head_dim = head_dim

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        input_pos: Tensor = args[0] if len(args) > 0 else kwargs["input_pos"]
        freqs_cis: Tensor = args[1] if len(args) > 1 else kwargs["freqs_cis"]
        attn_mask: Tensor | None = args[2] if len(args) > 2 else kwargs.get("attn_mask")

        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)

        # rope
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (b, s, h, d) -> (b, h, s, d)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        # cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # gqa
        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        dropout_p = self.dropout_p if self.training else 0

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


Attention = GroupedQueryAttentionWithRoPEAndCache

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dnn.modeling.pos_encoding import apply_rotary_emb, precompute_freqs_cis


class KVCache(nn.Module):
    """Key-Value Cache for Transformer Decoder.

    Total memory usage:
        2 * n_layers * max_batch_size * n_heads * head_dim * max_seq_length * dtype_size.
        Typical 3:1 ratio between kv cache and model parameters.

    Example:
        OPT-30B: 2 * 48 * 128 * 7168 * 1024 * 2 = 180 GB vs 60 GB model parameters.
    """

    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float16):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype

        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.cache_shape = cache_shape

        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Update the cache with new key and value tensors.

        Args:
            input_pos (Tensor): Input positions. Shape (S).
        """
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, input_pos] = k
        v_out[:, :, input_pos] = v

        return k_out, v_out


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0,
    is_causal: bool = False,
    scale: float | None = None,
):
    """Scaled Dot Product Attention.

    Masked impleemntation of scaled dot product attention with dropout.

    Args:
        query (Tensor): Query tensor. Shape (batch_size, num_heads, seq_len, embed_dim).
        key (Tensor): Key tensor. Shape (batch_size, num_heads, seq_len, embed_dim).
        value (Tensor): Value tensor. Shape (batch_size, num_heads, seq_len, embed_dim).
        attn_mask (Tensor, optional): Attention mask. Defaults to None. Shape (batch_size, num_heads, seq_len, seq_len).
        dropout_p (float, optional): Dropout probability. Defaults to 0.
        is_causal (bool, optional): Causal attention. Defaults to False.
        scale (float, optional): Scale factor. Defaults to None.

    Returns:
        Tensor: Scaled dot product attention tensor. Shape (batch_size, num_heads, seq_len, embed_dim).
    """
    b, h, n, d = query.size()
    if scale is None:
        scale = d**-0.5
    if is_causal:
        attn_mask = torch.tril(torch.ones(query.size(-2), query.size(-2)))
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        # attn += mask
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
    attn = F.dropout(attn, p=dropout_p)
    y = torch.matmul(attn, value)
    return y


class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0,
        bias: bool = True,
        kv_cache: Optional[KVCache] = None,
        sdpa: Callable = F.scaled_dot_product_attention,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.bias = bias
        self.kv_cache = kv_cache
        self.sdpa = sdpa
        self.is_causal = is_causal
        # if self.is_causal:
        #     self.register_buffer("attn_mask", torch.tril(torch.ones(1024, 1024))

        self.qkv_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.qkv_dim)

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, input_pos: Optional[Tensor] = None) -> Tensor:
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

        x = self.sdpa(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal, dropout_p=dropout_p, scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        x = self.out_proj(x)
        x = self.dropout(x)  # necessary?
        return x


MultiheadAttention = MHA


class MultiQueryAttention(nn.Module):
    """Multi Query Attention mechanism.

    Is identical except that the different heads share a single set of keys and value
    """

    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        dropout_p: float = 0,
        bias: bool = True,
        sdpa: Callable = F.scaled_dot_product_attention,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = 1
        self.dropout_p = dropout_p
        self.bias = bias
        self.sdpa = sdpa
        self.is_causal = is_causal

        self.head_dim = embed_dim // num_q_heads
        self.scale = 1 / math.sqrt(embed_dim)

        self.wq = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, 1 * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, 1 * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, -1, 1, self.head_dim).expand(-1, -1, self.num_q_heads, -1)
        v = self.wv(x).view(bsz, -1, 1, self.head_dim).expand(-1, -1, self.num_q_heads, -1)

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        x = self.sdpa(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal, dropout_p=dropout_p, scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout_p: float = 0,
        bias: bool = True,
        sdpa: Callable = F.scaled_dot_product_attention,
        is_causal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_p = dropout_p
        self.bias = bias
        self.sdpa = sdpa
        self.is_causal = is_causal

        self.head_dim = embed_dim // num_q_heads
        self.scale = 1 / math.sqrt(embed_dim)
        self.kv_repeats = num_q_heads // num_kv_heads

        self.wq = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
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

        x = self.sdpa(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal, dropout_p=dropout_p, scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


GQA = GroupedQueryAttention


class GroupedQueryAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout_p: float = 0,
        bias: bool = True,
        sdpa: Callable = F.scaled_dot_product_attention,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_p = dropout_p
        self.bias = bias
        self.sdpa = sdpa
        self.is_causal = is_causal

        self.head_dim = embed_dim // num_q_heads
        self.scale = 1 / math.sqrt(embed_dim)
        self.kv_repeats = num_q_heads // num_kv_heads

        self.wq = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor, freqs_cis: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        x = self.sdpa(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal, dropout_p=dropout_p, scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


class GroupedQueryAttentionWithRoPEAndCache(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        kv_cache: Optional[KVCache] = None,
        dropout_p: float = 0,
        bias: bool = False,
        sdpa: Callable = F.scaled_dot_product_attention,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_p = dropout_p
        self.bias = bias
        self.sdpa = sdpa
        self.is_causal = is_causal
        self.kv_cache = kv_cache

        head_dim = embed_dim // num_q_heads
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(embed_dim)
        self.kv_repeats = num_q_heads // num_kv_heads

        self.wq = nn.Linear(embed_dim, num_q_heads * head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_q_heads * head_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)

        # rope
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (b, s, h, d) -> (b, h, s, d)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # gqa
        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        x = self.sdpa(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal, dropout_p=dropout_p, scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


Attention = GroupedQueryAttentionWithRoPEAndCache


class CrossAttention(nn.Module):
    """Cross Attention mechanism.

    - Cross atenntion is an attention mechanism in Transformer architecture that mixes two different embedding sequences
    - One of the sequences defines the output length as it plays a role of a query input.
    - The other sequence then produces key and value input.
    """

    def __init__(self, embed_dim: int, num_heads=12, qkv_bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.qkv_dim = embed_dim // num_heads
        self.scale = self.qkv_dim**-0.5

        self.wq = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.wkv = nn.Linear(embed_dim, int(embed_dim * 2), bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: Tensor, x: Tensor) -> Tensor:
        """
        q comes from the decoder, x comes from the encoder hidden states.
        """
        B, n, C = q.shape
        q = self.wq(q).reshape(B, n, self.num_heads, self.qkv_dim).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.wkv(x).reshape(B, N, 2, self.num_heads, self.qkv_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        with torch.backends.cuda.sdp_kernel():
            q = scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)

        q = q.transpose(1, 2).reshape(B, n, C)
        return q

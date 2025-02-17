import path
from typing import Callable
import torch
import torch.nn.functional as F
from torch import Tensor, nn


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

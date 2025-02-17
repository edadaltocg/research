import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        device=None,
        dtype=None,
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

        self.wq = nn.Linear(
            embed_dim,
            num_q_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.wk = nn.Linear(embed_dim, 1 * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.wv = nn.Linear(embed_dim, 1 * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, -1, 1, self.head_dim).expand(-1, -1, self.num_q_heads, -1)
        v = self.wv(x).view(bsz, -1, 1, self.head_dim).expand(-1, -1, self.num_q_heads, -1)

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

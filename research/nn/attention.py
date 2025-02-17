import torch
import torch.nn.functional as F
from torch import Tensor, nn


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

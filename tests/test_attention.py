import pysnooper
import torch
import torch.nn.functional as F

from dnn.modeling.attention import (
    GroupedQueryAttentionWithRoPEAndCache,
    KVCache,
    scaled_dot_product_attention,
)
from dnn.modeling.pos_encoding import precompute_freqs_cis
from utils import benchmark_torch_function_in_microseconds, seed_all

seed_all(42)


def test_causal_attention_mask():
    seq_len = 10
    attn_mask = torch.tril(torch.ones(seq_len, seq_len))
    print("Attn mask", attn_mask)
    attn = torch.randn(1, 8, seq_len, seq_len)
    attn = attn.masked_fill(attn_mask == 0, float("-inf"))
    assert attn.size() == (1, 8, seq_len, seq_len)
    assert attn[-1, -1, 0, -1] == float("-inf")
    assert attn[-1, -1, -1, -1] != float("-inf")


def test_scaled_dot_product_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lets define the hyper-parameters of our input
    batch_size = 16
    max_sequence_len = 1024
    num_heads = 8
    embed_dimension = 128
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    query, key, value = (
        torch.randn(
            batch_size,
            num_heads,
            max_sequence_len,
            embed_dimension,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            batch_size,
            num_heads,
            max_sequence_len,
            embed_dimension,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            batch_size,
            num_heads,
            max_sequence_len,
            embed_dimension,
            device=device,
            dtype=dtype,
        ),
    )

    scale = embed_dimension**-0.5
    y1 = F.scaled_dot_product_attention(
        query, key, value, is_causal=False, dropout_p=0.0, scale=scale
    )
    assert y1.shape == (batch_size, num_heads, max_sequence_len, embed_dimension)

    y2 = scaled_dot_product_attention(
        query, key, value, is_causal=False, dropout_p=0.0, scale=scale
    )
    assert y2.shape == (batch_size, num_heads, max_sequence_len, embed_dimension)

    assert torch.allclose(y1, y2, atol=1e-4), (y1 - y2).abs().max()

    fns = [
        ("torch", F.scaled_dot_product_attention),
        ("ours", scaled_dot_product_attention),
    ]
    for f in fns:
        with torch.no_grad():
            for is_causal in [True, False]:
                print(
                    "Causal" if is_causal else "Non-Causal",
                    f[0],
                    round(
                        benchmark_torch_function_in_microseconds(
                            f[1], query, key, value, is_causal=False, dropout_p=0.0
                        ),
                        0,
                    ),
                    "μs",
                )

    y1 = F.scaled_dot_product_attention(
        query, key, value, is_causal=True, dropout_p=0.0
    )
    y2 = scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
    assert torch.allclose(y1, y2, atol=1e-4)


@pysnooper.snoop()
def test_group_query_attention_with_rope_and_kv_cache():
    embed_dim = 16
    num_q_heads = 4
    head_dim = embed_dim // num_q_heads
    num_kv_heads = 1
    batch_size = 1
    max_seq_len = 24
    kv_cache = KVCache(
        batch_size, max_seq_len, num_kv_heads, head_dim, dtype=torch.float32
    )

    layer = GroupedQueryAttentionWithRoPEAndCache(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        kv_cache=kv_cache,
        bias=False,
    )

    prompt_len = max_seq_len // 2
    x = torch.randn(batch_size, prompt_len, embed_dim)
    pos = torch.arange(0, x.shape[1])
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
    freqs_cis_ = freqs_cis[pos]

    # analogous to prefill
    with torch.no_grad():
        x = layer(x, pos, freqs_cis=freqs_cis_)
    assert x.size() == (batch_size, prompt_len, embed_dim)

    cache = layer.kv_cache.k_cache
    print(cache)

    next_x = torch.randn(batch_size, 1, embed_dim)
    pos = torch.tensor([prompt_len])
    freqs_cis_ = freqs_cis[pos]

    # analogous to decoding
    with torch.no_grad():
        x = layer(next_x, pos, freqs_cis=freqs_cis_)
    print(cache)

    assert x.size() == (batch_size, 1, embed_dim)

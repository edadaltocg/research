import torch.nn.functional as F
from dnn.modeling.attention import MHA
from dnn.modeling.dense import MLP
from dnn.modeling.pos_encoding import EmbeddingPositionalEncoding
from dnn.modeling.transformer import (
    LayerNorm,
    VanillaTransformerEncoder,
    VanillaTransformerEncoderLayer,
)
from torch import nn


class GPT(VanillaTransformerEncoder):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__(
            embedding=nn.Embedding(vocab_size, embed_dim),
            pos_encoding=EmbeddingPositionalEncoding(embed_dim, max_len=max_seq_len),
            encoder_layer=VanillaTransformerEncoderLayer(
                mlp=MLP(embed_dim, 4 * embed_dim, activation=F.gelu),
                attn_mechanism=MHA(embed_dim, num_heads),
                norm_layer1=LayerNorm(embed_dim),
                norm_layer2=LayerNorm(embed_dim),
                norm_first=True,
            ),
            num_encoder_layers=num_layers,
            norm=LayerNorm(embed_dim),
            head=nn.Linear(embed_dim, vocab_size, bias=False),
        )


gpt2_tiny_config = dict(
    vocab_size=50257,
    embed_dim=384,
    max_seq_len=512,
    num_heads=6,
    num_layers=6,
)

gpt2_config = dict(
    vocab_size=50257,
    embed_dim=768,
    max_seq_len=1024,
    num_heads=12,
    num_layers=12,
)
gpt_medium_config = dict(
    vocab_size=50257,
    embed_dim=1024,
    max_seq_len=1024,
    num_heads=16,
    num_layers=24,
)
gpt2_large_config = dict(
    vocab_size=50257,
    embed_dim=1280,
    max_seq_len=1024,
    num_heads=20,
    num_layers=36,
)
gpt_xl_config = dict(
    vocab_size=50257,
    embed_dim=1600,
    max_seq_len=1024,
    num_heads=25,
    num_layers=48,
)

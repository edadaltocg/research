"""
Models to learn how to play sudoku.

1. Strategy #1: EBMs
    f(x, y) = E(x, y)
    Generate and verify prediction
    Input is an incomplete board.
    Output of the generation is a complete board.
    Verification takes incomplete and complete board and gives its energy.

    Architectures:
        MLP -- do not scale with board rank
        Siamese networks -- prone to collapse?
        Convolutional networks
        Transformers -- encoder decoder architecture? Sparse attention (constraints)?
2. Graph neural networks
    Difficulty: adjacency matrix do not scale
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn

from ebm_sudoku_solver.game.sudoku import suggest_board

# TODO: lora for fine tuning
# TODO: patch equivalent tokens? Each possible block is a token. vocab=rank**2!
# TODO: reset parameters for layers


class _sdpsa(nn.Module):
    """Scaled Dot-Product Self Attention

    O(N^2 D)"""

    def __init__(
        self,
        hidden_dim: int,
        dim_v: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        # scale parameter
        # gradients of softmax becomes exponentially small for large dimensions
        # the elements of the matrices are gaussian r.v. with mean 0 and var 1
        # the product has var D
        # so we scale by the standard deviation, or the square root of D
        self.dim = hidden_dim

        # learnable weights
        self.qw = nn.Linear(bias=False, in_features=hidden_dim, out_features=hidden_dim)
        self.kw = nn.Linear(bias=False, in_features=hidden_dim, out_features=hidden_dim)
        self.vw = nn.Linear(bias=False, in_features=hidden_dim, out_features=dim_v)

        # regularization
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        # x: NxD
        # weighted sum
        # sum_m anm xm
        q = self.qw(x)
        k: Tensor = self.kw(x)
        v = self.vw(x)
        a = torch.softmax(torch.div(torch.matmul(q, k.transpose(2, 1)), math.sqrt(self.dim)), dim=-1)  # NxN
        a = self.dropout(a)
        return torch.matmul(a, v)  # NxD_v


class _msdpsa(_sdpsa):
    """Masked Scaled Dot-Product Self Attention

    Or causal attention"""

    MASK_VALUE = -torch.inf

    def forward(self, x: Tensor) -> Tensor:
        # x: NxD
        # weighted sum
        # sum_m anm xm
        device, dtype = x.device, x.dtype
        b, n, d = x.shape
        q: Tensor = self.qw(x)
        k: Tensor = self.kw(x)
        v: Tensor = self.vw(x)

        # causal mask
        m = torch.ones(n, n, device=device, dtype=dtype).triu(diagonal=1)
        m = m.masked_fill(m == 1, self.MASK_VALUE).unsqueeze(0)

        scores = torch.div(torch.matmul(q, k.transpose(2, 1)), math.sqrt(self.dim))
        a = torch.softmax(m + scores, dim=-1)
        a = self.dropout(a)
        return torch.matmul(a, v)  # NxD_v


class _sdpca(nn.Module):
    """Scaled Dot-Product Cross-Attention

    O(N^2 D)"""

    def __init__(
        self,
        hidden_dim: int,
        dim_v: int,
        dropout_p: float = 0.1,
    ):
        super().__init__(dim=hidden_dim, dim_v=dim_v)
        # scale parameter
        # gradients of softmax becomes exponentially small for large dimensions
        # the elements of the matrices are gaussian r.v. with mean 0 and var 1
        # the product has var D
        # so we scale by the standard deviation, or the square root of D
        self.dim = hidden_dim

        # learnable weights
        self.qw = nn.Linear(bias=False, in_features=hidden_dim, out_features=hidden_dim)
        self.kw = nn.Linear(bias=False, in_features=hidden_dim, out_features=hidden_dim)
        self.vw = nn.Linear(bias=False, in_features=hidden_dim, out_features=dim_v)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        # x: NxD
        # weighted sum
        # sum_m anm xm
        q = self.qw(x)
        k: Tensor = self.kw(z)
        v = self.vw(z)
        a = torch.softmax(torch.div(torch.matmul(q, k.transpose(2, 1)), math.sqrt(self.dim)), dim=-1)  # NxN
        a = self.dropout(a)
        return torch.matmul(a, v)  # NxD_v


class _sudoku_msdpsa(nn.Module):
    # TODO: implement sudoku attention
    """I want to have an attention mechanism that makes training more efficient by
    constraining through a mask where attention is not needed."""


class _mha(nn.Module):
    """Multi-head Attention"""

    # TODO: create appropriate protocol
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout_p: float = 0.1,
        *,
        attention_op: type[_sdpsa] | type[_msdpsa] = _sdpsa,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = hidden_dim
        self.dim_v = hidden_dim // num_heads

        # learnable weights
        self.sdpsa = nn.ModuleList([attention_op(hidden_dim, self.dim_v, dropout_p) for _ in range(num_heads)])
        self.ow = nn.Linear(bias=False, in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: NxD
        x_ = [h(x) for h in self.sdpsa]
        x = torch.cat(x_, dim=-1)
        return self.ow(x)  # NxD


class _mmha(_mha):
    """Masked Multi-head Attention"""

    def __init__(self, num_heads: int, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__(num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p, attention_op=_msdpsa)


class _mhca(nn.Module):
    """Multi-head Cross-Attention"""

    def __init__(self, num_heads: int, hidden_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = hidden_dim
        self.dim_v = hidden_dim // num_heads
        attention_op = _sdpsa

        # learnable weights
        self.sdpsa = nn.ModuleList([attention_op(hidden_dim, self.dim_v, dropout_p) for _ in range(num_heads)])
        self.ow = nn.Linear(bias=False, in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: NxD
        x_ = [h(x) for h in self.sdpsa]
        x = torch.cat(x_, dim=-1)
        return self.ow(x)  # NxD


class LayerNorm(nn.Module):
    """
    $$
    y = \frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \beta
    $$
    """

    def __init__(
        self,
        normalized_shape: int | tuple,
        eps: float = 1e-05,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        return self.weight * (x - mean) / std + self.bias

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _layernorm(nn.Module):
    # TODO: Why does it improve training efficiency?
    """LayerNorm layer

    improves training efficiency.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Multi-layer perceptron.

    O(NxD^2)"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _tokenizer(data) -> Tensor:
    """Transforms the raw data into a tensor of dimension NxD.

    Common special tokens: <start>, <end>, <pad>, <class>, <mask>
    """

    raise NotImplementedError


def _embedding(vocab_size: int, embedding_dim: int) -> nn.Module:
    """Transforms the tokenized data into a tensor of dimension NxD."""
    embedding = nn.Embedding(vocab_size, embedding_dim)
    return embedding


def _spe(x: Tensor, *, max_seq_len: int, embed_dim: int, l: int = 10_000) -> Tensor:
    """Sinusoidal Positional Encoding

    Two randomly chosen uncorrelated vectors tend to be nearly orthogonal in spaces of
    high dimensionality.

    The vector values all lie in (-1, 1).

    The network should be able to attend to relative positions with these operations.
    """
    # l is a constant
    # n is the position
    # i is the embedding index
    b, n, d = x.shape
    device, dtype = x.device, x.dtype

    # TODO: buffer this
    seq_pos = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)  # (n, 1)
    hid_pos = torch.arange(0, embed_dim, 2, device=device, dtype=dtype).unsqueeze(0)  #  (1, d/2)

    f = torch.pow(l, torch.div(hid_pos, d))  # (1, d/2)

    # i is even
    s = torch.sin(seq_pos / f)  # (n, d/2)
    # i is odd
    c = torch.cos(seq_pos / f)  # (n, d/2)

    pe = torch.zeros(n, d, device=device, dtype=dtype)
    pe[:, 0::2] = s  # even
    pe[:, 1::2] = c  # odd
    return x + pe[:n].unsqueeze(0)


class _sinusoidal_positional_encoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Two randomly chosen uncorrelated vectors tend to be nearly orthogonal in
    spaces of high dimensionality. The sin/cos pairing (sharing frequencies)
    lets the network attend to *relative* positions via a linear transform
    between offsets. All values lie in [-1, 1].

    The table is precomputed once in fp32 and stored in a buffer, so it moves
    with the module (.to/.cuda/.half) and is not a trainable parameter.
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 4096, l: int = 10_000):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        # n is the position, i is the embedding index, l is a constant
        seq_pos = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        hid_pos = torch.arange(0, embed_dim, 2).unsqueeze(0)  # (1, d/2)
        f = torch.pow(l, hid_pos / embed_dim)  # (1, d/2)

        s = torch.sin(seq_pos / f)  # (max_seq_len, d/2)
        c = torch.cos(seq_pos / f)  # (max_seq_len, d/2)

        pe = torch.zeros(max_seq_len, embed_dim)  # (max_seq_len, d)
        pe[:, 0::2] = s  # even
        pe[:, 1::2] = c  # odd

        # (1, max_seq_len, d) for broadcasting over the batch
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, D) -> (B, S, D)"""
        b, n, d = x.shape
        # cast to input dtype at use (keeps mixed-precision safe)
        return x + self.pe[:, :n].to(dtype=x.dtype)


def _sudoku_pe():
    """I want to have equivariance on specific permutation operations
    that are corresponding to sudoku symmetries."""


class _transformer_block_attention_stage(nn.Module):
    """PostNormMHA"""

    # TODO: Why pre-norm improve efficiency?
    # TODO: use proper protocols
    # mix token information
    def __init__(
        self, num_heads: int, hidden_dim: int, dropout_p: float, *, attention_layer: type[_mha] | type[_mmha]
    ) -> None:
        super().__init__()
        self.attention = attention_layer(num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p)
        self.layernorm = _layernorm(hidden_dim=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.attention(x)
        # residual connection
        x = x + x_
        x = self.layernorm(x)
        return x


class _transformer_block_mlp_stage(nn.Module):
    """PostNormMLP"""

    # apply idependently to every token
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = _mlp(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.layernorm = _layernorm(hidden_dim=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.mlp(x)
        # residual connection
        x = x + x_
        x = self.layernorm(x)
        return x


def _lsm(hidden_dim: int, vocab_size: int) -> nn.Module:
    """Linear softmax"""
    return nn.Sequential(nn.Linear(hidden_dim, vocab_size, bias=False), nn.Softmax(dim=-1))


def _transformer_block(
    num_heads: int, hidden_dim: int, dropout_p: float, attention_layer: type[_mha] | type[_mmha]
) -> nn.Sequential:
    """
    Attention mechanism
    LayerNorm
    MLP
    Residual connections
    """
    return nn.Sequential(
        _transformer_block_attention_stage(
            num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p, attention_layer=attention_layer
        ),
        _transformer_block_mlp_stage(hidden_dim=hidden_dim),
    )


class _cross_attention_transformer_block(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout_p: float):
        self.stage_1 = _transformer_block_attention_stage(
            num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p, attention_layer=_mmha
        )
        self.cross_attention = _mhca(num_heads=num_heads, hidden_dim=hidden_dim)
        self.layernorm = _layernorm(hidden_dim=hidden_dim)
        self.stage_3 = _transformer_block_mlp_stage(hidden_dim=hidden_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = self.stage_1(x)
        x_ = self.cross_attention(x, z)
        x = x + x_
        x = self.layernorm(x)
        x = self.stage_3(x)
        return x


def _transformer(
    vocab_size: int,
    max_seq_len: int,
    num_blocks: int,
    num_heads: int,
    hidden_dim: int,
    dropout_p: float,
    attention_layer: type[_mha] | type[_mmha],
) -> nn.Sequential:
    """
    Embedding
    Positional encoding
    Transformer blocks x L
    Head
    """
    layers = OrderedDict()
    layers["embedding"] = _embedding(vocab_size=vocab_size, embedding_dim=hidden_dim)
    layers["pos_encoding"] = _sinusoidal_positional_encoding(embed_dim=hidden_dim, max_seq_len=max_seq_len)
    for i in range(num_blocks):
        layers[f"block_{i}"] = _transformer_block(
            num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p, attention_layer=attention_layer
        )
    layers["head"] = _lsm(hidden_dim=hidden_dim, vocab_size=vocab_size)

    return nn.Sequential(layers)


def _decoder_transformer(
    vocab_size: int, max_seq_len: int, num_blocks: int, num_heads: int, hidden_dim: int, dropout_p: float
) -> nn.Sequential:
    return _transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        attention_layer=_mmha,
    )


def _encoder_transformer(
    vocab_size: int, max_seq_len: int, num_blocks: int, num_heads: int, hidden_dim: int, dropout_p: float
) -> nn.Sequential:
    layers = OrderedDict()
    layers["embedding"] = _embedding(vocab_size=vocab_size, embedding_dim=hidden_dim)
    layers["pos_encoding"] = _sinusoidal_positional_encoding(embed_dim=hidden_dim, max_seq_len=max_seq_len)
    for i in range(num_blocks):
        layers[f"block_{i}"] = _transformer_block(
            num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p, attention_layer=_mha
        )

    return nn.Sequential(layers)


class _cross_attention_decoder_transformer(nn.Module):
    def __init__(
        self, vocab_size: int, max_seq_len: int, num_blocks: int, num_heads: int, hidden_dim: int, dropout_p: float
    ):
        super().__init__()

        self.embedding = _embedding(vocab_size=vocab_size, embedding_dim=hidden_dim)
        self.pe = _sinusoidal_positional_encoding(embed_dim=hidden_dim, max_seq_len=max_seq_len)

        layers = OrderedDict()
        for i in range(num_blocks):
            layers[f"block_{i}"] = _cross_attention_transformer_block(
                num_heads=num_heads, hidden_dim=hidden_dim, dropout_p=dropout_p
            )
        self.layers = nn.Sequential(layers)

        self.head = _lsm(hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.pe(x)
        for h in self.layers:
            x = h(x, z)
        x = self.head(x)
        return x


class EncoderDecoderTransformer(nn.Module):
    """Encoder Decoder Transformer

    Sequence to sequence modeling"""

    def __init__(
        self, vocab_size: int, max_seq_len: int, num_blocks: int, num_heads: int, hidden_dim: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.encoder = _encoder_transformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        )
        self.decoder = _cross_attention_decoder_transformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.decoder(y, z)


class SudokuTowerModel(nn.Module):
    def __init__(self, board_rank: int, embed_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        # accounts for empty cell
        vocab_size = board_rank**2 + 1
        # doesn't scale well with board_rank
        seq_len = board_rank**4

        self.embedding = _embedding(vocab_size, embedding_dim=embed_dim)
        self.pe = _sinusoidal_positional_encoding(max_seq_len=seq_len, embed_dim=hidden_dim)
        self.mlp = _mlp(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # b, w, w, e
        logger.debug(f"{x.shape=}")
        x = x.flatten(1, 2)  # b, w*w, e
        logger.debug(f"{x.shape=}")
        x = x.transpose(1, 2)  # b, e, w*w
        logger.debug(f"{x.shape=}")
        x = self.pe(x)  # b, e, h
        logger.debug(f"{x.shape=}")
        # pooling
        x = x.mean(1)  # b, h
        logger.debug(f"{x.shape=}")
        x = self.mlp(x)
        logger.debug(f"{x.shape=}")
        return x


class EBMSudokuVerifierModel(nn.Module):
    """Verifier and representation model."""

    def __init__(self, board_rank: int, embed_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.tower = SudokuTowerModel(board_rank=board_rank, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.tower(x)
        y = self.tower(y)

        return -torch.cosine_similarity(x, y, dim=-1)


class SudokuGenerativeModel(nn.Module):
    """Generative model"""

    def __init__(self, input_dim: int, board_rank: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seq_len = board_rank**4
        self.vocab_size = board_rank**2

    def forward(self, x: Tensor) -> Tensor:
        # TODO:
        return x


@dataclass()
class ModelConfig:
    num_blocks: int
    hidden_dim: int
    num_heads: int


class SudokuModelArchitectures(Enum):
    SMALL = ...
    BASE = ...
    LARGE = ...


if __name__ == "__main__":
    model = EBMSudokuVerifierModel(board_rank=3)
    x, y = suggest_board()
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    incompatibility = model(x, y)
    print(f"{incompatibility=}")
    print(f"{incompatibility.shape=}")

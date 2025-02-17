r"""The Transformer.

Self Attention

$$
    \text{softmax}\left(\frac{\mathbf{Q K}^T}{\sqrt{d_q}} \circ \mathbf{M}\right) V
$$

Self-attention is invariant to permutations and changes in the number of input points.

Complexity: $O(n^2 d)$.

Configuration:

    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str | ((Tensor) -> Tensor) = F.relu,
    custom_encoder: Any | None = None,
    custom_decoder: Any | None = None,
    layer_norm_eps: float = 0.00001,
    batch_first: bool = False,
    norm_first: bool = False,
    bias: bool = True,
    device: Unknown | None = None,
    dtype: Unknown | None = None

References:
    [1] V. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
        Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances
        in Neural Information Processing Systems, 2017, pp. 6000–6010.
"""

import math
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy

import torch
from torch import Tensor, nn


def init_weights(
    module: nn.Linear | nn.Embedding,
    init_fn: str,
    std=0.02,
    d_model: int = 768,
    n_layers: int = 1,
    std_factor: float = 1.0,
) -> None:
    """
    Initialize weights of a linear or embedding module.
    """
    if init_fn == "normal":
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif init_fn == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif init_fn == "fan_in":
        std = std_factor / math.sqrt(d_model)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    else:
        raise NotImplementedError(init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if init_fn == "normal" and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * n_layers))


class CLSToken(nn.Module):
    def __init__(self, embed_dim):
        """
        Initialize the CLS Token module.

        Args:
            embed_dim (int): The dimension of the embedding space.
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cls_token)

    def forward(self, x: Tensor) -> Tensor:
        # Get the batch size from the input tensor
        batch_size = x.size(0)

        # Expand the cls_token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend the cls_token to the input sequence
        x = torch.cat((cls_tokens, x), dim=1)

        return x


class VanillaTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attn_mechanism: nn.Module,
        mlp: nn.Module,
        norm_layer1: nn.Module,
        norm_layer2: nn.Module,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm2 = norm_layer2

        self.norm_first = norm_first
        self.attn = attn_mechanism
        self.norm1 = norm_layer1
        self.mlp = mlp

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.norm_first:
            # self-attention
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
            # mlp
            x = x + self.mlp(self.norm2(x))
        else:
            # self-attention
            x = self.norm1(x + self.attn(x, attn_mask=attn_mask))
            # mlp
            x = self.norm2(x + self.mlp(x))
        return x


class VanillaTransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        pos_encoding: nn.Module | Callable,
        encoder_layer: nn.Module,
        num_encoder_layers=6,
        dropout_p=0.1,
        cls_token: nn.Module | None = None,
        norm: nn.Module | None = None,
        pool: Callable | None = None,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.cls_token = cls_token
        self.pos_encoding = pos_encoding
        self.num_encoder_layers = num_encoder_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_encoder_layers)])
        self.norm = norm
        self.pool = pool
        self.head = head

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        # (b, s, ...) -> (b, s, d)
        x = self.embedding(x)
        if self.cls_token is not None:
            # (b, s, d) -> (b, s + 1, d)
            x = self.cls_token(x)
        # (b, s, d) -> (b, s, d) + (b, s, d)
        x += self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.head is not None:
            # (b, s|1, d) -> (b, s|1, k)
            x = self.head(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        mlp: nn.Module,
        norm_layer1: nn.Module,
        norm_layer2: nn.Module,
        norm_layer3: nn.Module,
        norm_first: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.self_attn = self_attention
        self.cross_attn = cross_attention
        self.mlp = mlp
        self.norm1 = norm_layer1
        self.norm2 = norm_layer2
        self.norm3 = norm_layer3
        self.norm_first = norm_first

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        """Decoder layer forward pass.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        """
        if self.norm_first:
            # self-attention
            tgt = tgt + self.attn(self.norm1(tgt), attn_mask=tgt_mask)
            # cross-attention
            tgt = tgt + self.cross_attn(self.norm2(tgt), memory, attn_mask=memory_mask)
            # mlp
            tgt = tgt + self.mlp(self.norm2(tgt))
        else:
            # self-attention
            tgt = self.norm1(tgt + self.attn(tgt, attn_mask=tgt_mask))
            # cross-attention
            tgt = self.norm2(tgt + self.cross_attn(tgt, memory, attn_mask=memory_mask))
            # mlp
            tgt = self.norm2(tgt + self.mlp(tgt))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        pos_encoding: nn.Module,
        embedding: nn.Module,
        decoder_layer: nn.Module,
        num_decoder_layers: int,
        dropout_p: float,
        norm: nn.Module | None = None,
        head: nn.Module | None = None,
        pool: Callable | None = None,
    ) -> None:
        super().__init__()
        self.pos_encoding = pos_encoding
        self.embedding = embedding
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        layers: OrderedDict[str, nn.Module] = OrderedDict({
            f"{i}": deepcopy(decoder_layer) for i in range(num_decoder_layers)
        })
        self.layers = nn.Sequential(layers)

        self.dropout = nn.Dropout(dropout_p)
        self.norm = norm
        self.pool = pool
        self.head = head

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """Transformer decoder forward pass.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
        """
        tgt = self.embedding(tgt)
        tgt += self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        tgt = self.layers(tgt, memory)

        if self.norm is not None:
            tgt = self.norm(tgt)
        if self.pool is not None:
            tgt = self.pool(tgt)
        if self.head is not None:
            tgt = self.head(tgt)

        return tgt


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Transformer encoder decoder architecture forward pass.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
        """
        memory = self.encoder(src, src_mask)
        outputs = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return outputs

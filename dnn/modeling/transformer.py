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
from copy import deepcopy
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | tuple,
        eps=1e-05,
        bias=True,
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attn_mechanism: nn.Module,
        mlp: nn.Module,
        norm_layer1: nn.Module,
        norm_layer2: nn.Module,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.attn = attn_mechanism
        self.mlp = mlp
        self.norm1 = norm_layer1
        self.norm2 = norm_layer2
        self.norm_first = norm_first

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, input_pos: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            # self-attention
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask, input_pos=input_pos)
            # mlp
            x = x + self.mlp(self.norm2(x))
        else:
            # self-attention
            x = self.norm1(x + self.attn(x, attn_mask=attn_mask, input_pos=input_pos))
            # mlp
            x = self.norm2(x + self.mlp(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        pos_encoding: Union[nn.Module, Callable],
        encoder_layer: nn.Module,
        num_encoder_layers=6,
        dropout_p=0.1,
        cls_token: Optional[Tensor] = None,
        norm: Optional[nn.Module] = None,
        pool: Optional[Callable] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.cls_token = cls_token
        self.pos_encoding = pos_encoding
        self.num_encoder_layers = num_encoder_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)
        # self.layers = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            {f"{i}": deepcopy(encoder_layer) for i in range(num_encoder_layers)}
        )
        self.layers = nn.Sequential(layers)
        self.norm = norm
        self.pool = pool
        self.head = head

        for name, module in self.named_modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        # (b, s, ...) -> (b, s, d)
        x = self.embedding(x)
        if self.cls_token is not None:
            # (b, s, d) -> (b, s + 1, d)
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        # (b, s, d) -> (b, s, d) + (b, s, d)
        x += self.pos_encoding(x)
        x = self.dropout(x)
        # for layer in self.layers:
        #     x = layer(x)
        x = self.layers(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.head is not None:
            # (b, s|1, d) -> (b, s|1, k)
            x = self.head(x)
        return x


# When refered to a "Transformer" it usually means the encoder part.
Transformer = TransformerEncoder


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
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
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
        norm: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        pool: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.pos_encoding = pos_encoding
        self.embedding = embedding
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            {f"{i}": deepcopy(decoder_layer) for i in range(num_decoder_layers)}
        )
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
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
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


def init_weights(
    module: Union[nn.Linear, nn.Embedding],
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

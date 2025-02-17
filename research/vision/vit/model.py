from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from research.nn.layers.mha import MHA
from research.nn.layers.pos_encoding import FixedPositionalEncoding
from research.nn.models.mlp import MLP
from research.nn.models.transformer import (
    CLSToken,
    VanillaTransformerEncoder,
    VanillaTransformerEncoderLayer,
)
from research.vision.vit.patch_embedding import PatchEmbedding


class VisionTransformerEncoder(VanillaTransformerEncoder):
    """Vision Transformer.

    Transformer _encoder_ for vision representation.

    References:
        [1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
            T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,
            et al., "An image is worth 16x16 words: Transformers for image
            recognition at scale," arXiv preprint arXiv:2010.11929, 2020.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout_p: float,
        attn_dropout_p: float,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout_p = dropout_p
        self.attn_dropout_p = attn_dropout_p

        self.seq_len = (image_size // patch_size) ** 2 + 1  # +1 for cls_token

        def pool(x: Tensor) -> Tensor:
            return x[:, 0]  # cls token

        super().__init__(
            embedding=PatchEmbedding(patch_size, in_channels, hidden_dim),
            pos_encoding=FixedPositionalEncoding(self.seq_len, hidden_dim),
            encoder_layer=VanillaTransformerEncoderLayer(
                attn_mechanism=MHA(
                    hidden_dim,
                    num_heads,
                    attn_dropout_p,
                    bias=True,
                    sdpa=F.scaled_dot_product_attention,
                ),
                mlp=MLP(hidden_dim, mlp_dim, dropout_p, activation=F.gelu),
                norm_layer1=nn.LayerNorm(hidden_dim, eps=1e-6),
                norm_layer2=nn.LayerNorm(hidden_dim, eps=1e-6),
                norm_first=True,
            ),
            num_encoder_layers=num_layers,
            dropout_p=dropout_p,
            cls_token=CLSToken(hidden_dim),
            norm=nn.LayerNorm(hidden_dim, eps=1e-6),
            pool=pool,
        )


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        *,
        patch_size=16,
        in_channels=3,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout_p=0.1,
        attn_dropout_p=0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout_p = dropout_p
        self.attn_dropout_p = attn_dropout_p

        self.encoder = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout_p=dropout_p,
            attn_dropout_p=attn_dropout_p,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x


ViT = VisionTransformer


def convert_from_pytorch_vit(source_state_dict):
    """Convert from source_state_dict to target_state_dict."""
    tgt_state_dict = OrderedDict()
    for k, v in source_state_dict.items():
        new_key = (
            k.replace("encoder_layer_", "")
            .replace("encoder.", "")
            .replace("conv_proj", "embedding.proj")
            .replace("pos_embedding", "pos_encoding.pe")
            .replace("class_token", "cls_token")
            .replace("ln", "norm")
            .replace("self_attention", "attn")
            .replace("linear_", "linear")
            .replace("norm_", "norm")
            .replace("in_proj_", "qkv_proj.")
            .replace("heads.", "")
        )

        tgt_state_dict[new_key] = v
    return tgt_state_dict


def get_hf_vit(name: str = "google/vit-base-patch16-224"):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name)
    return processor, model

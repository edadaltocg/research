from typing import Any

# Define a common configuration for all ViT models
_vit_common_config: dict[str, Any] = {
    "in_channels": 3,
    "dropout_p": 0.1,
    "attn_dropout_p": 0.1,
}

# Define specific configurations for different ViT model sizes
vit_tiny_config: dict[str, Any] = {
    **_vit_common_config,
    "num_layers": 12,
    "num_heads": 3,
    "hidden_dim": 192,
    "mlp_dim": 768,
}

vit_small_config: dict[str, Any] = {
    **_vit_common_config,
    "num_layers": 12,
    "num_heads": 6,
    "hidden_dim": 384,
    "mlp_dim": 1536,
}

vit_base_config: dict[str, Any] = {
    **_vit_common_config,
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
}

vit_b_16_config: dict[str, float | int] = {
    **vit_base_config,
    "patch_size": 16,
}

vit_large_config: dict[str, float | int] = {
    **_vit_common_config,
}

vit_huge_config: dict[str, float | int] = {
    **_vit_common_config,
}

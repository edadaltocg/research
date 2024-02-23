from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, n_h * n_w, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class LinearPatchEmbedding(nn.Module):
    def __init__(self, patch_height: int, patch_width: int, patch_dim: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

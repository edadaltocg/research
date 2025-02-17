import torch
from torch import Tensor, nn
import torch.nn.functional as F


# 2D


class UNet2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, conv_kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor):
        return self.block(x)


class UNet2DEncoderBlock(UNet2DBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv_kernel_size, dropout_p)
        self.pooling = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x: Tensor):
        x = super().forward(x)
        x = self.pooling(x)
        return x


class UNet2DBridge(UNet2DBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        up_conv_kernel_size: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv_kernel_size, dropout_p)
        self.up_conv = nn.ConvTranspose2d(
            out_channels,
            out_channels // 2,
            up_conv_kernel_size,
            stride=up_conv_kernel_size,
        )

    def forward(self, x: Tensor):
        x = super().forward(x)
        x = self.up_conv(x)
        return x


class UNet2DDecoderBlock(UNet2DBridge):
    def forward(self, encoder_feature: Tensor, x: Tensor):  # pyright: ignore
        # x: BCHW
        b, c, h, w = x.shape
        bf, cf, hf, wf = encoder_feature.shape
        diffW = wf - w
        diffH = hf - h
        # crop
        x = nn.functional.pad(
            x,
            (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2),
            "constant",
            0,
        )  # from last to first

        # concatenate
        x = torch.concatenate([encoder_feature, x], dim=1)  # CWH
        x = super().forward(x)
        return x


class UNet2DHead(UNet2DBlock):
    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv_kernel_size, dropout_p)
        self.conv = nn.Conv2d(out_channels, n_classes, 1)

    def forward(self, encoder_feature: Tensor, x: Tensor):  # pyright: ignore
        # x: BCHW
        _, _, h, w = x.shape
        _, _, hf, wf = encoder_feature.shape
        diffW = wf - w
        diffH = hf - h
        # crop
        x = nn.functional.pad(
            x,
            (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2),
            "constant",
            0,
        )  # from last to first

        # concatenate
        x = torch.concatenate([encoder_feature, x], dim=1)  # CWH
        x = super().forward(x)
        x = self.conv(x)
        return x


class UNet2D(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        in_channels: int = 3,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        decoder_channels: tuple[int, int, int, int] = (1024, 512, 256, 128),
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList([
            UNet2DEncoderBlock(
                in_channels,
                encoder_channels[0],
                conv_kernel_size,
                pool_kernel_size,
                dropout_p,
            )
        ])
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                UNet2DEncoderBlock(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    conv_kernel_size,
                    pool_kernel_size,
                    dropout_p,
                )
            )

        self.bridge = UNet2DBridge(
            encoder_channels[-1],
            decoder_channels[0],
            conv_kernel_size,
            pool_kernel_size,
            dropout_p,
        )

        self.decoders = nn.ModuleList([
            UNet2DDecoderBlock(
                decoder_channels[i],
                decoder_channels[i + 1],
                conv_kernel_size,
                pool_kernel_size,
                dropout_p,
            )
            for i in range(len(decoder_channels) - 1)
        ])

        self.head = UNet2DHead(
            n_classes,
            decoder_channels[-1],
            encoder_channels[0],
            conv_kernel_size,
            dropout_p,
        )

    def forward(self, x: Tensor):
        _, _, h, w = x.shape

        encoders_features = []
        for enc in self.encoders:
            x = enc(x)
            encoders_features.append(x)
        encoders_features.reverse()

        x = self.bridge(x)

        for dec, encoder_feature in zip(self.decoders, encoders_features):
            x = dec(encoder_feature, x)

        x = self.head(encoders_features[-1], x)
        x = F.interpolate(x, (h, w), mode="bilinear")
        return x


# 3D


class UNet3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, conv_kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, conv_kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor):
        return self.block(x)


class UNet3DEncoderBlock(UNet3DBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv_kernel_size, dropout_p)
        self.pooling = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x: Tensor):
        x = super().forward(x)
        x = self.pooling(x)
        return x


class UNet3DBridge(UNet3DBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        up_conv_kernel_size: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv_kernel_size, dropout_p)
        self.up_conv = nn.ConvTranspose3d(
            out_channels,
            out_channels // 2,
            up_conv_kernel_size,
            stride=up_conv_kernel_size,
        )

    def forward(self, x: Tensor):
        x = super().forward(x)
        x = self.up_conv(x)
        return x


class UNet3DDecoderBlock(UNet3DBridge):
    def forward(self, encoder_feature: Tensor, x: Tensor):  # pyright: ignore
        # x: B, C, D, H, W
        b, c, d, h, w = x.shape
        _, _, d_e, h_e, w_e = encoder_feature.shape
        diffD = d_e - d
        diffH = h_e - h
        diffW = w_e - w
        # pad
        x = nn.functional.pad(
            x,
            (
                diffW // 2,
                diffW - diffW // 2,
                diffH // 2,
                diffH - diffH // 2,
                diffD // 2,
                diffD - diffD // 2,
            ),
            "constant",
            0,
        )  # Pads (W_left, W_right, H_top, H_bottom, D_front, D_back)
        # concatenate
        x = torch.cat([encoder_feature, x], dim=1)  # concatenate along channels
        x = super().forward(x)
        return x


class UNet3DHead(UNet3DBlock):
    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv_kernel_size, dropout_p)
        self.conv = nn.Conv3d(out_channels, n_classes, 1)

    def forward(self, encoder_feature: Tensor, x: Tensor):  # pyright: ignore
        # x: B, C, D, H, W
        _, _, d, h, w = x.shape
        _, _, d_e, h_e, w_e = encoder_feature.shape
        diffD = d_e - d
        diffH = h_e - h
        diffW = w_e - w
        # pad
        x = nn.functional.pad(
            x,
            (
                diffW // 2,
                diffW - diffW // 2,
                diffH // 2,
                diffH - diffH // 2,
                diffD // 2,
                diffD - diffD // 2,
            ),
            "constant",
            0,
        )  # Pads (W_left, W_right, H_top, H_bottom, D_front, D_back)
        # concatenate
        x = torch.cat([encoder_feature, x], dim=1)  # concatenate along channels
        x = super().forward(x)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        in_channels: int = 1,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        encoder_channels: tuple = (64, 128, 256, 512),
        decoder_channels: tuple = (1024, 512, 256, 128),
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList([
            UNet3DEncoderBlock(
                in_channels,
                encoder_channels[0],
                conv_kernel_size,
                pool_kernel_size,
                dropout_p,
            )
        ])
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                UNet3DEncoderBlock(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    conv_kernel_size,
                    pool_kernel_size,
                    dropout_p,
                )
            )

        self.bridge = UNet3DBridge(
            encoder_channels[-1],
            decoder_channels[0],
            conv_kernel_size,
            pool_kernel_size,
            dropout_p,
        )

        self.decoders = nn.ModuleList([
            UNet3DDecoderBlock(
                decoder_channels[i],
                decoder_channels[i + 1],
                conv_kernel_size,
                pool_kernel_size,
                dropout_p,
            )
            for i in range(len(decoder_channels) - 1)
        ])

        self.head = UNet3DHead(
            n_classes,
            decoder_channels[-1],
            encoder_channels[0],
            conv_kernel_size,
            dropout_p,
        )

    def forward(self, x: Tensor):
        _, _, d, h, w = x.shape

        encoders_features = []
        for enc in self.encoders:
            x = enc(x)
            encoders_features.append(x)
        encoders_features.reverse()

        x = self.bridge(x)

        for dec, encoder_feature in zip(self.decoders, encoders_features):
            x = dec(encoder_feature, x)

        x = self.head(encoders_features[-1], x)
        x = F.interpolate(x, size=(d, h, w), mode="trilinear", align_corners=False)
        return x

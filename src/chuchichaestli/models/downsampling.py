# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Downsampling modules for 1, 2, and 3D inputs."""

import torch
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, DIM_TO_POOL_MAP, DOWNSAMPLE_MODE
from chuchichaestli.utils import partialclass
from typing import Literal


__all__ = [
    "Downsample",
    "DownsampleInterpolate",
    "DownsampleUnshuffle",
    "Pool",
    "MaxPool",
    "AdaptiveMaxPool",
    "AvgPool",
    "AdaptiveAvgPool",
    "DOWNSAMPLE_FUNCTIONS",
]


DownsampleTypes = Literal["Downsample", "DownsampleInterpolate", "DownsampleUnshuffle"]


class Downsample(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs."""

    def __init__(self, dimensions: int, num_channels: int, **kwargs):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 2)
        kwargs.setdefault("padding", 1)
        self.conv = conv_cls(num_channels, num_channels, **kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        return self.conv(x)


class DownsampleInterpolate(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs implemented with interpolation.

    Note: In the U-Net architecture, downsampling by interpolation is not commonly used.
    """

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        factor: int | None = None,
        antialias: bool = False,
        with_conv: bool = False,
        **kwargs,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.num_channels = num_channels
        self.factor = factor if factor is not None else 2
        self.align_corners = False
        self.antialias = antialias
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        if with_conv:
            self.conv = conv_cls(num_channels, num_channels, **kwargs)

    @property
    def mode(self) -> Literal["linear", "bilinear", "trilinear", "nearest"]:
        """Interpolation mode."""
        return DOWNSAMPLE_MODE.get(self.dimensions, "nearest")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        spatial_dims = x.shape[2:]
        output_dims = [s // self.factor for s in spatial_dims]
        x = F.interpolate(
            x,
            size=output_dims,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        if hasattr(self, "conv"):
            x = self.conv(x)
        return x


class DownsampleUnshuffle(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs implemented with pixel shuffling."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        factor: int | None = None,
        **kwargs,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.factor = factor if factor is not None else 2
        r2 = self.factor**2
        self.group_size = in_channels * r2 // out_channels
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        self.conv = conv_cls(in_channels, out_channels // r2, **kwargs)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.factor)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        h = self.pixel_unshuffle(self.conv(x))
        shortcut = self.pixel_unshuffle(x)
        shortcut = shortcut.unflatten(1, (-1, self.group_size)).mean(dim=2)
        return h + shortcut


class Pool(nn.Module):
    """Max/avg (optionally adaptive) pooling layer for 1D, 2D, and 3D inputs."""

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        average: bool = False,
        adaptive: bool = False,
        **kwargs,
    ):
        """Initialize the pooling layer (default: max pooling)."""
        super().__init__()
        pool_type = "MaxPool"
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 2)
        kwargs.setdefault("padding", 1)
        if average:
            pool_type = "AvgPool"
        if adaptive:
            pool_type = "Adaptive" + pool_type
            out = kwargs.get("output_size", None)
            kwargs = {"output_size": out if out is not None else (1,) * dimensions}
        pool_cls = DIM_TO_POOL_MAP[dimensions][pool_type]
        self.pool = pool_cls(**kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the pooling layer."""
        return self.pool(x)


MaxPool = partialclass("MaxPool", Pool, average=False, adaptive=False)
AdaptiveMaxPool = partialclass("AdaptiveMaxPool", Pool, adaptive=True)
AvgPool = partialclass("AvgPool", Pool, average=True)
AdaptiveAvgPool = partialclass("AdaptiveAvgPool", Pool, average=True, adaptive=True)


DOWNSAMPLE_FUNCTIONS = {
    "Downsample": Downsample,
    "DownsampleInterpolate": DownsampleInterpolate,
    "DownsampleUnshuffle": DownsampleUnshuffle,
    "MaxPool": MaxPool,
    "AdaptiveMaxPool": AdaptiveMaxPool,
    "AvgPool": AvgPool,
    "AdaptiveAvgPool": AdaptiveAvgPool,
}

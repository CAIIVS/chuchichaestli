# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Upsampling modules for 1, 2, and 3D inputs."""

import torch
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, DIM_TO_CONVT_MAP, UPSAMPLE_MODE
from typing import Literal


__all__ = [
    "Upsample",
    "UpsampleInterpolate",
    "UpsampleShuffle",
]

UpsampleTypes = Literal["Upsample", "UpsampleInterpolate", "UpsampleShuffle"]


class Upsample(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs."""

    def __init__(self, dimensions: int, num_channels: int):
        """Initialize the upsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONVT_MAP[dimensions]
        self.conv = conv_cls(
            num_channels, num_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, _t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        return self.conv(x)


class UpsampleInterpolate(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with interpolation."""

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        factor: int | None = None,
        antialias: bool = False,
        with_conv: bool = True,
        **kwargs,
    ):
        """Initialize the upsampling layer."""
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
        return UPSAMPLE_MODE.get(self.dimensions, "nearest")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        spatial_dims = x.shape[2:]
        output_dims = [s * self.factor for s in spatial_dims]
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


class UpsampleShuffle(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with pixel shuffling."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        factor: int | None = None,
        **kwargs,
    ):
        """Initialize the upsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.factor = factor if factor is not None else 2
        r2 = self.factor**2
        self.repeats = out_channels * r2 // in_channels
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        self.conv = conv_cls(in_channels, out_channels * r2, **kwargs)
        self.pixel_shuffle = nn.PixelShuffle(self.factor)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        h = self.pixel_shuffle(self.conv(x))
        shortcut = x.repeat_interleave(
            self.repeats, dim=1, output_size=x.shape[1] * self.repeats
        )
        shortcut = self.pixel_shuffle(shortcut)
        return h + shortcut


UPSAMPLE_FUNCTIONS = {
    "Upsample": Upsample,
    "UpsampleInterpolate": UpsampleInterpolate,
    "UpsampleShuffle": UpsampleShuffle,
}

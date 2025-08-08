# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Downsampling modules for 1, 2, and 3D inputs."""

import torch
from torch.nn import Module
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, DOWNSAMPLE_MODE
from typing import Literal


class Downsample(Module):
    """Downsampling layer for 1D, 2D, and 3D inputs."""

    def __init__(self, dimensions: int, num_channels: int):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.conv = conv_cls(
            num_channels, num_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        return self.conv(x)


class DownsampleInterpolate(Module):
    """Downsampling layer for 1D, 2D, and 3D inputs implemented with interpolation.

    Note: In the U-Net architecture, downsampling by interpolation is not commonly used.
    """

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        factor: int | None = None,
        antialias: bool = False,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        self.dimensions = dimensions
        self.num_channels = num_channels
        self.factor = factor if factor is not None else 2
        self.align_corners = False
        self.antialias = antialias

    @property
    def mode(self) -> Literal["linear", "bilinear", "trilinear", "nearest"]:
        """Interpolation mode."""
        return DOWNSAMPLE_MODE.get(self.dimensions, "nearest")

    def forward(self, x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        spatial_dims = x.shape[2:]
        output_dims = [s // self.factor for s in spatial_dims]
        return F.interpolate(
            x,
            size=output_dims,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )


DOWNSAMPLE_FUNCTIONS = {
    "Downsample": Downsample,
    "DownsampleInterpolate": DownsampleInterpolate,
}

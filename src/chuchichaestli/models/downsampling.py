# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Downsampling modules for 1, 2, and 3D inputs."""

import torch
from torch import nn

from chuchichaestli.models.maps import DIM_TO_CONV_MAP


class Downsample(nn.Module):
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


class DownsampleInterpolate(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs implemented with interpolation."""

    def __init__(self, _dimensions: int, num_channels: int):
        """Initialize the downsampling layer."""
        raise NotImplementedError("DownsampleInterpolate is not implemented yet.")

    def forward(self, x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        raise NotImplementedError("DownsampleInterpolate is not implemented yet.")

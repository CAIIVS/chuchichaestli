# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Upsampling modules for 1, 2, and 3D inputs."""

import torch
from torch import nn

from chuchichaestli.models.maps import DIM_TO_CONVT_MAP


class Upsample(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs."""

    def __init__(self, dimensions: int, num_channels: int):
        """Initialize the upsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONVT_MAP[dimensions]
        self.conv = conv_cls(
            num_channels, num_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        return self.conv(x)


class UpsampleInterpolate(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with interpolation."""

    def __init__(self, _dimensions: int, num_channels: int):
        """Initialize the upsampling layer."""
        raise NotImplementedError("UpsampleInterpolate is not implemented yet.")

    def forward(self, x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        raise NotImplementedError("UpsampleInterpolate is not implemented yet.")

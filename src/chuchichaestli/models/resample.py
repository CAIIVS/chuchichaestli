# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resampling modules shared by the down- and upsampling paths."""

import torch
from torch import nn

from chuchichaestli.models.maps import DIM_TO_CONV_MAP


__all__ = ["ChannelResample"]


class ChannelResample(nn.Module):
    """Pointwise resampling of the channel axis for 1D, 2D, and 3D inputs.

    Leaves the spatial axes untouched, which turns a hierarchical model into a
    constant-resolution one that still widens between levels. It scales neither
    up nor down spatially, so the same layer serves both halves of a model and
    is registered as a sampler in either direction.
    """

    changes_channels = True
    factor = 1

    def __init__(self, dimensions: int, in_channels: int, out_channels: int, **kwargs):
        """Initialize the channel resampling layer.

        Args:
            dimensions: Number of spatial dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kwargs: Additional keyword arguments for the convolution.
        """
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        kwargs.setdefault("kernel_size", 1)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", 0)
        self.conv = conv_cls(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the channel resampling layer."""
        return self.conv(x)

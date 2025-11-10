# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalization modules for neural networks."""

import torch
from torch import nn
from typing import Literal


__all__ = ["Norm"]


NormTypes = Literal["group", "instance", "batch", "adabatch", "rms", "layer"]


class Norm(nn.Module):
    """Normalization layer implementation."""

    def __init__(
        self,
        dimensions: int,
        norm_type: NormTypes,
        channels: int,
        num_groups: int,
        **kwargs,
    ):
        """Initialize the normalization layer."""
        super().__init__()
        self.ntype = norm_type
        self.norm: nn.Module
        match norm_type:
            case "group":
                self.norm = nn.GroupNorm(num_groups, channels)
            case "instance" if dimensions == 1:
                self.norm = nn.InstanceNorm1d(channels)
            case "instance" if dimensions == 2:
                self.norm = nn.InstanceNorm2d(channels)
            case "instance" if dimensions == 3:
                self.norm = nn.InstanceNorm3d(channels)
            case "batch" if dimensions == 1:
                self.norm = nn.BatchNorm1d(channels, **kwargs)
            case "batch" if dimensions == 2:
                self.norm = nn.BatchNorm2d(channels, **kwargs)
            case "batch" if dimensions == 3:
                self.norm = nn.BatchNorm3d(channels, **kwargs)
            case "rms":
                self.norm = nn.RMSNorm(channels)
            case "layer":
                self.norm = nn.LayerNorm(channels)
            case "adabatch":
                self.norm = AdaptiveBatchNorm(dimensions, channels, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass through the normalization layer."""
        if self.ntype in ("rms", "layer"):
            return self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return self.norm(x)


class AdaptiveBatchNorm(nn.Module):
    """Adaptive BN implementation with two additional parameters."""

    def __init__(
        self,
        dimensions: int,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """Constructor."""
        super().__init__()
        self.bn = Norm(
            dimensions,
            "batch",
            channels,
            0,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )
        self.a = nn.Parameter(torch.FloatTensor(1, 1, *((1,) * dimensions)))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, *((1,) * dimensions)))

    def forward(self, x):
        """Adaptive BN with two additional parameters `a` and `b`.

        Return:
          a * x + b * bn(x)
        """
        return self.a * x + self.b * self.bn(x)

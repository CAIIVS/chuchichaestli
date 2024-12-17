"""Normalization modules for neural networks.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

import torch
from torch import nn


__all__ = ["Norm"]


class Norm(nn.Module):
    """Normalization layer implementation."""

    def __init__(
        self, dimensions: int, norm_type: str, channels: int, num_groups: int, **kwargs
    ):
        """Initialize the normalization layer."""
        super().__init__()
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
            case "abatch":
                self.norm = AdaptiveBatchNorm(dimensions, channels, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass through the normalization layer."""
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

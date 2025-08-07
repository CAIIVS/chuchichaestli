"""Upsampling layers 1, 2, and 3D inputs.

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
from torch.nn import Module
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONVT_MAP, UPSAMPLE_MODE
from typing import Literal


class Upsample(Module):
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


class UpsampleInterpolate(Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with interpolation."""

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        factor: int | None = None,
        antialias: bool = False,
    ):
        """Initialize the upsampling layer."""
        self.dimensions = dimensions
        self.num_channels = num_channels
        self.factor = factor if factor is not None else 2
        self.align_corners = False
        self.antialias = antialias

    @property
    def mode(self) -> Literal["linear", "bilinear", "trilinear", "nearest"]:
        """Interpolation mode."""
        return UPSAMPLE_MODE.get(self.dimensions, "nearest")

    def forward(self, x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        spatial_dims = x.shape[2:]
        output_dims = [s * self.factor for s in spatial_dims]
        return F.interpolate(
            x,
            size=output_dims,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )


UPSAMPLE_FUNCTIONS = {
    "Upsample": Upsample,
    "UpsampleInterpolate": UpsampleInterpolate,
}

"""Blocks for adversarial models.

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

from functools import partial
import torch
from torch import nn
from chuchichaestli.models.resnet import Norm
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from collections.abc import Callable


class ConvDownBlock(nn.Module):
    """Convolutional downsampling block."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        act_fn: str | None = None,
        norm_type: str | None = None,
        num_groups: int = 32,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        **kwargs,
    ):
        """Initialize the convolutional downsampling block."""
        super().__init__()
        self.norm: nn.Module | None = None
        self.act: nn.Module | None = None
        if norm_type is not None:
            self.norm = Norm(dimensions, norm_type, in_channels, num_groups)
        if act_fn is not None:
            self.act = ACTIVATION_FUNCTIONS[act_fn]()
        self.conv = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )

    def forward(self, x: torch.Tensor, _t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the convolutional downsampling block."""
        h = x
        h = self.norm(h) if self.norm is not None else h
        h = self.act(h) if self.act is not None else h
        return self.conv(h)


ConvBlock = partial(ConvDownBlock, stride=1)
NormConvBlock = partial(ConvDownBlock, norm_type="batch", stride=1)
NormConvDownBlock = partial(ConvDownBlock, norm_type="batch")
ActConvBlock = partial(ConvDownBlock, act_fn="leakyrelu,0.2", stride=1)
ActConvDownBlock = partial(ConvDownBlock, act_fn="leakyrelu,0.2")
NormActConvBlock = partial(ConvDownBlock, norm_type="batch", act_fn="leakyrelu,0.2", stride=1)
NormActConvDownBlock = partial(ConvDownBlock, norm_type="batch", act_fn="leakyrelu,0.2")


BLOCK_MAP: dict[str, Callable] = {
    "ConvBlock": ConvBlock,
    "ConvDownBlock": ConvDownBlock,
    "NormConvBlock": NormConvBlock,
    "NormConvDownBlock": NormConvDownBlock,
    "ActConvBlock": ActConvBlock,
    "ActConvDownBlock": ActConvDownBlock,
    "NormActConvBlock": NormActConvBlock,
    "NormActConvDownBlock": NormActConvDownBlock,
}

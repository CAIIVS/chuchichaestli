"""UNet building blocks.

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

from chuchichaestli.models.resnet import ResidualBlock


class DownBlock(nn.Module):
    """Down block for UNet."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = True,
        time_channels: int = 32,
        n_res_blocks: int = 1,
        res_args: dict = {},
    ):
        """Initialize the down block."""
        super().__init__()
        self.res_blocks = nn.ModuleList(
            ResidualBlock(
                dimensions,
                in_channels,
                out_channels,
                time_embedding,
                time_channels,
                **res_args,
            )
            for _ in range(n_res_blocks)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the down block."""
        for res_block in self.res_blocks:
            x = res_block(x, t)
        return x


class MidBlock(nn.Module):
    """Mid block for UNet."""

    def __init__(
        self,
        dimensions: int,
        channels: int,
        time_embedding: bool = True,
        time_channels: int = 32,
        n_res_blocks: int = 1,
        res_args: dict = {},
    ):
        """Initialize the mid block."""
        super().__init__()
        self.res_blocks = nn.ModuleList(
            ResidualBlock(
                dimensions,
                channels,
                channels,
                time_embedding,
                time_channels,
                **res_args,
            )
            for _ in range(n_res_blocks)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mid block."""
        for res_block in self.res_blocks:
            x = res_block(x, t)
        return x


class UpBlock(nn.Module):
    """Up block for UNet."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = True,
        time_channels: int = 32,
        n_res_blocks: int = 1,
        res_args: dict = {},
    ):
        """Initialize the up block."""
        super().__init__()
        self.res_blocks = nn.ModuleList(
            ResidualBlock(
                dimensions,
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_embedding,
                time_channels,
                **res_args,
            )
            for i in range(n_res_blocks)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the up block."""
        for res_block in self.res_blocks:
            x = res_block(x, t)
        return x

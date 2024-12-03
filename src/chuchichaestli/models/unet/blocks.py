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

from functools import partial

import torch
from torch import nn

from chuchichaestli.models.resnet import ResidualBlock
from chuchichaestli.models.attention import ATTENTION_MAP


class DownBlock(nn.Module):
    """Down block for UNet."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = True,
        time_channels: int = 32,
        res_args: dict = {},
        attention: str | None = None,
        attn_args: dict = {},
    ):
        """Initialize the down block."""
        super().__init__()
        self.res_block = ResidualBlock(
            dimensions,
            in_channels,
            out_channels,
            time_embedding,
            time_channels,
            **res_args,
        )

        match ATTENTION_MAP.get(attention, None):
            case "self_attention":
                self.attn = ATTENTION_MAP[attention](in_channels, **attn_args)
            case _:
                self.attn = None

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the down block."""
        x = self.attn(x, None) if self.attn else x
        x = self.res_block(x, t)
        return x


class MidBlock(nn.Module):
    """Mid block for UNet."""

    def __init__(
        self,
        dimensions: int,
        channels: int,
        time_embedding: bool = True,
        time_channels: int = 32,
        res_args: dict = {},
        attention: str | None = None,
        attn_args: dict = {},
    ):
        """Initialize the mid block."""
        super().__init__()
        self.res_block = ResidualBlock(
            dimensions,
            channels,
            channels,
            time_embedding,
            time_channels,
            **res_args,
        )
        match ATTENTION_MAP.get(attention, None):
            case "self_attention":
                self.attn = ATTENTION_MAP[attention](channels, **attn_args)
            case _:
                self.attn = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mid block."""
        x = self.attn(x, None) if self.attn else x
        x = self.res_block(x, t)
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
        res_args: dict = {},
        attention: str | None = None,
        attn_args: dict = {},
        skip_connection_action: str | None = None,
    ):
        """Initialize the up block."""
        super().__init__()
        self.skip_connection_action = skip_connection_action
        if skip_connection_action == "concat":
            self.res_block = ResidualBlock(
                dimensions,
                in_channels + out_channels,
                out_channels,
                time_embedding,
                time_channels,
                **res_args,
            )
        elif skip_connection_action in ["avg", "add", None]:
            self.res_block = ResidualBlock(
                dimensions,
                in_channels,
                out_channels,
                time_embedding,
                time_channels,
                **res_args,
            )
        else:
            raise ValueError(
                f"Invalid skip connection action: {skip_connection_action}"
            )

        match ATTENTION_MAP.get(attention, None):
            case "self_attention":
                self.attn = ATTENTION_MAP[attention](in_channels, **attn_args)
            case "attention_gate":
                self.attn = ATTENTION_MAP[attention](
                    in_channels, out_channels, **attn_args
                )
            case _:
                self.attn = None

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the up block."""
        x = self.attn(x, h) if self.attn else x
        if self.skip_connection_action == "avg":
            replication_factor = x.shape[1] // h.shape[1]
            h = h.repeat(
                (1, replication_factor) + (1,) * len(x.shape[2:])
            )  # Repeat channels
            xh = (x + h) / 2
        elif self.skip_connection_action == "add":
            replication_factor = x.shape[1] // h.shape[1]
            h = h.repeat(
                (1, replication_factor) + (1,) * len(x.shape[2:])
            )  # Repeat channels
            xh = x + h
        elif self.skip_connection_action == "concat":
            xh = torch.cat([x, h], dim=1)
        else:
            xh = x
        x = self.res_block(xh, t)
        return x


AttnDownBlock = partial(DownBlock, attention="self_attention")
AttnMidBlock = partial(MidBlock, attention="self_attention")
AttnUpBlock = partial(UpBlock, attention="self_attention")
AttnGateUpBlock = partial(UpBlock, attention="attention_gate")

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""ResNet block implementation."""

import torch
from torch import nn

from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import Norm


class ResidualBlock(nn.Module):
    """Residual block implementation."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool,
        time_channels: int,
        res_groups: int = 32,
        res_act_fn: str = "silu",
        res_dropout: float = 0.1,
        res_norm_type: str = "group",
        res_kernel_size: int = 3,
    ):
        """Initialize the residual block."""
        super().__init__()
        act_cls = ACTIVATION_FUNCTIONS[res_act_fn]
        conv_cls = DIM_TO_CONV_MAP[dimensions]

        self.dimensions = dimensions

        self.norm1 = Norm(dimensions, res_norm_type, in_channels, res_groups)
        self.act1 = act_cls()
        self.conv1 = conv_cls(
            in_channels, out_channels, kernel_size=res_kernel_size, padding="same"
        )

        self.norm2 = Norm(dimensions, res_norm_type, out_channels, res_groups)
        self.act2 = act_cls()
        self.conv2 = conv_cls(
            out_channels, out_channels, kernel_size=res_kernel_size, padding="same"
        )

        self.shortcut = (
            conv_cls(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.time_embedding = time_embedding
        if time_embedding:
            self.time_proj = nn.Linear(time_channels, out_channels)
            self.time_act = act_cls()

        self.dropout = nn.Dropout(res_dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the residual block."""
        hh = self.conv1(self.act1(self.norm1(x)))
        idx = [slice(None), slice(None)] + [None] * self.dimensions
        if self.time_embedding:
            hh += self.time_proj(self.time_act(t))[idx]
        hh = self.conv2(self.dropout(self.act2(self.norm2(hh))))

        return hh + self.shortcut(x)

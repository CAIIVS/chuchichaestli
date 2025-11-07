# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Encoder modules for autoencoders (AEs)."""

from torch import nn

from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.blocks import (
    BLOCK_MAP,
    AutoencoderDownBlockTypes,
    AutoencoderMidBlockTypes,
    EncoderOutBlockTypes,
)
from chuchichaestli.models.downsampling import DOWNSAMPLE_FUNCTIONS
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import NormTypes
from typing import Literal
from collections.abc import Sequence


class Encoder(nn.Module):
    """Flexible encoder implementation for autoencoders."""

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 64,
        out_channels: int = 1,
        down_block_types: Sequence[AutoencoderDownBlockTypes] = (
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 2),
        num_layers_per_block: int = 2,
        mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        out_block_type: EncoderOutBlockTypes = "EncoderOutBlock",
        downsample_type: Literal["Downsample", "DownsampleInterpolate"] = "Downsample",
        act_fn: ActivationTypes = "silu",
        norm_type: NormTypes = "group",
        num_groups: int = 8,
        kernel_size: int = 3,
        res_args: dict = {},
        attn_args: dict = {},
        double_z: bool = True,
    ) -> None:
        """Constructor.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels.
            n_channels: Number of channels in the hidden layer.
            out_channels: Number of output channels (latent space; doubled if `double_z`).
            down_block_types: Type of down blocks to use for each level.
            block_out_channel_mults: Multiplier for output channels of each block.
            num_layers_per_block: Number of blocks per level (blocks are repeated if `>1`).
            mid_block_types: Type of blocks to use before the output.
            out_block_type: Type of block for output (latent space).
            downsample_type: Type of downsampling block
                (see `chuchichaestli.models.downsampling` for details).
            act_fn: Activation function for the output layers
                (see `chuchichaestli.models.activations` for details).
            norm_type: Normalization type for the output layer.
            num_groups: Number of groups for normalization in the output layer.
            kernel_size: Kernel size for the output convolution.
            res_args: Arguments for residual blocks.
            attn_args: Arguments for attention blocks.
            double_z (bool): Whether to double the latent space.
        """
        super().__init__()

        downsample_cls = DOWNSAMPLE_FUNCTIONS[downsample_type]
        n_mults = len(block_out_channel_mults)

        self.conv_in = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            n_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.down_blocks = nn.ModuleList()
        ins = n_channels
        for i in range(n_mults):
            outs = ins * block_out_channel_mults[i]

            for _ in range(num_layers_per_block):
                down_block = BLOCK_MAP[down_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    res_args=res_args,
                    attn_args=attn_args,
                )
                self.down_blocks.append(down_block)
                ins = outs

            if i < n_mults - 1:
                self.down_blocks.append(downsample_cls(dimensions, ins))

        self.mid_blocks = nn.ModuleList([])
        for mid_block_type in mid_block_types:
            mid_block = BLOCK_MAP[mid_block_type](
                dimensions=dimensions,
                channels=outs,
                res_args=res_args,
                attn_args=attn_args,
            )
            self.mid_blocks.append(mid_block)

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.out_block = BLOCK_MAP[out_block_type](
            dimensions=dimensions,
            in_channels=outs,
            out_channels=conv_out_channels,
            act_fn=act_fn,
            norm_type=norm_type,
            num_groups=num_groups,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.mid_blocks:
            x = block(x)
        x = self.out_block(x)
        return x

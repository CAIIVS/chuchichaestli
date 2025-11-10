# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoder modules for autoencoders."""

import torch
from torch import nn
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.blocks import (
    BLOCK_MAP,
    NormActConvBlock,
    AutoencoderUpBlockTypes,
    AutoencoderMidBlockTypes,
    DecoderInBlockTypes,
)
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.upsampling import UPSAMPLE_FUNCTIONS, UpsampleTypes
from chuchichaestli.utils import prod
from collections.abc import Sequence


class Decoder(nn.Module):
    """Flexible decoder implementation for autoencoders."""

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 512,
        out_channels: int = 1,
        in_block_type: DecoderInBlockTypes = "DecoderInBlock",
        mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        up_block_types: Sequence[AutoencoderUpBlockTypes] = (
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 2),
        num_layers_per_block: int | Sequence[int] = 3,
        upsample_type: UpsampleTypes = "UpsampleInterpolate",
        act_fn: ActivationTypes = "silu",
        norm_type: NormTypes = "group",
        num_groups: int = 8,
        kernel_size: int = 3,
        res_args: dict = {},
        attn_args: dict = {},
        in_shortcut: bool = False,
    ):
        """Decoder implementation.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels (latent space).
            n_channels: Number of channels for first block.
            out_channels: Number of output channels.
            in_block_type: Type of block for output (latent space).
            mid_block_types: Type of blocks to use before the output.
            up_block_types: Type of up blocks to use for each level.
            block_out_channel_mults: Multiplier for output channels of each block.
            num_layers_per_block: Number of blocks per level (blocks are repeated if `>1`).
            upsample_type: Type of upsampling block (see `chuchichaestli.models.upsampling` for details).
            act_fn: Activation function for the output layers
                (see `chuchichaestli.models.activations` for details).
            norm_type: Normalization type for the output layer.
            num_groups: Number of groups for normalization in the output layer.
            kernel_size: Kernel size for the output convolution.
            res_args: Arguments for residual blocks.
            attn_args: Arguments for attention blocks.
            in_shortcut: Whether to use a shortcut for the input block.
        """
        super().__init__()

        upsample_cls = UPSAMPLE_FUNCTIONS[upsample_type]
        if len(block_out_channel_mults) < len(up_block_types):
            block_out_channel_mults += (1,) * (
                len(up_block_types) - len(block_out_channel_mults)
            )
        elif len(block_out_channel_mults) > len(up_block_types):
            block_out_channel_mults = block_out_channel_mults[: len(up_block_types)]
        n_mults = len(block_out_channel_mults)
        self.channel_mults = prod(block_out_channel_mults)
        if isinstance(num_layers_per_block, int):
            num_layers_per_block = (num_layers_per_block,) * n_mults
        elif len(num_layers_per_block) < len(up_block_types):
            num_layers_per_block += (num_layers_per_block[-1],) * (
                len(up_block_types) - len(num_layers_per_block)
            )

        self.in_block = BLOCK_MAP[in_block_type](
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
        )

        self.mid_blocks = nn.ModuleList([])
        for mid_block_type in mid_block_types:
            mid_block = BLOCK_MAP[mid_block_type](
                dimensions=dimensions,
                channels=n_channels,
                res_args=res_args,
                attn_args=attn_args,
            )
            self.mid_blocks.append(mid_block)

        self.up_blocks = nn.ModuleList([])
        ins = n_channels
        for i in range(n_mults):
            outs = ins
            if upsample_type != "UpsampleShuffle":
                outs = int(ins // block_out_channel_mults[i])
            stage = nn.Sequential()
            for _ in range(num_layers_per_block[i]):
                up_block = BLOCK_MAP[up_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    res_args=res_args,
                    attn_args=attn_args,
                )
                stage.append(up_block)
                ins = outs
            self.up_blocks.append(stage)

            if i < n_mults - 1:
                if upsample_type != "UpsampleShuffle":
                    self.up_blocks.append(upsample_cls(dimensions, outs))
                else:
                    self.up_blocks.append(
                        upsample_cls(
                            dimensions,
                            ins,
                            outs := int(ins // block_out_channel_mults[i]),
                        )
                    )
                    ins = outs
        self.levels = (len(self.up_blocks) + 1) // 2
        self.out_block = NormActConvBlock(
            dimensions=dimensions,
            in_channels=outs,
            out_channels=out_channels,
            act_fn=act_fn,
            norm_type=norm_type,
            num_groups=num_groups,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.in_shortcut = in_shortcut
        self.shortcut_repeats = n_channels // in_channels

    @property
    def f(self) -> int:
        """Expansion factor of the decoder."""
        return 2 ** max(self.levels - 1, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decoding forward pass."""
        if self.in_shortcut:
            shortcut = z.repeat_interleave(
                self.shortcut_repeats,
                dim=1,
                output_size=z.shape[1] * self.shortcut_repeats,
            )
            z = self.in_block(z) + shortcut
        else:
            z = self.in_block(z)
        for block in self.mid_blocks:
            z = block(z)
        for block in self.up_blocks:
            z = block(z)
        z = self.out_block(z)
        return z

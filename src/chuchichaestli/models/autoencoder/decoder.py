# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoder modules for autoencoders."""

import torch
from torch import nn
from chuchichaestli.models.upsampling import UPSAMPLE_FUNCTIONS
from chuchichaestli.models.blocks import BLOCK_MAP, CONV_BLOCK_MAP
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from typing import Literal
from collections.abc import Sequence


class Decoder(nn.Module):
    """Flexible decoder implementation for autoencoders."""

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 512,
        out_channels: int = 1,
        up_block_types: Sequence[
            Literal[
                "AutoencoderUpBlock",
                "AutoencoderAttnUpBlock",
                "AutoencoderConvAttnUpBlock",
            ]
        ] = (
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 2),
        num_layers_per_block: int = 3,
        mid_block_types: Sequence[
            Literal[
                "AutoencoderMidBlock",
                "AttnAutoencoderMidBlock",
                "ConvAttnAutoencoderMidBlock",
            ]
        ] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        in_block_type: Literal[
            "DecoderInBlock", "VAEDecoderInBlock"
        ] = "DecoderInBlock",
        upsample_type: Literal["Upsample", "UpsampleInterpolate"] = "UpsampleInterpolate",
        res_args: dict = {},
        attn_args: dict = {},
        in_out_args: dict = {},
    ) -> None:
        """Decoder implementation.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels (latent space).
            n_channels: Number of channels for first block.
            out_channels: Number of output channels.
            up_block_types: Type of up blocks to use for each level.
            block_out_channel_mults: Multiplier for output channels of each block.
            num_layers_per_block: Number of blocks per level (blocks are repeated if `>1`).
            mid_block_types: Type of blocks to use before the output.
            in_block_type: Type of block for output (latent space).
            upsample_type: Type of upsampling block (see `chuchichaestli.models.upsampling` for details).
            res_args: Arguments for residual blocks.
            attn_args: Arguments for attention blocks.
            in_out_args: Arguments for input and output convolutions.
        """
        super().__init__()

        upsample_cls = UPSAMPLE_FUNCTIONS[upsample_type]
        n_mults = len(block_out_channel_mults)

        self.in_block = BLOCK_MAP[in_block_type](
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=n_channels,
            **in_out_args,
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
            outs = ins // block_out_channel_mults[i]
            for _ in range(num_layers_per_block):
                up_block = BLOCK_MAP[up_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    res_args=res_args,
                    attn_args=attn_args,
                )
                self.up_blocks.append(up_block)
                ins = outs

            if i < n_mults - 1:
                self.up_blocks.append(upsample_cls(dimensions, outs))

        self.conv_out = CONV_BLOCK_MAP["NormActConvBlock"](
            dimensions=dimensions,
            in_channels=outs,
            out_channels=out_channels,
            act_fn=in_out_args.get("act_fn", "silu"),
            norm_type=in_out_args.get("norm_type", "group"),
            num_groups=in_out_args.get("num_groups", 4),
            kernel_size=in_out_args.get("kernel_size", 3),
            stride=1,
            padding="same"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decoding forward pass."""
        z = self.in_block(z)
        for block in self.mid_blocks:
            z = block(z)
        for block in self.up_blocks:
            z = block(z)
        z = self.conv_out(z)
        return z

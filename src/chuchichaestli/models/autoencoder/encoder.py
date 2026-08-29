# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Encoder modules for autoencoders (AEs)."""

from torch import nn

from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.blocks import (
    ATTN_BLOCK_MAP,
    INTRA_BLOCK_ATTN_ARGS,
    BLOCK_MAP,
    AutoencoderDownBlockTypes,
    AutoencoderMidBlockTypes,
    EncoderOutBlockTypes,
)
from chuchichaestli.models.downsampling import DOWNSAMPLE_FUNCTIONS, DownsampleTypes
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, require_cls
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.utils import broadcast, broadcast_kwargs, prod
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
        num_layers_per_block: int | Sequence[int] = 2,
        mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        out_block_type: EncoderOutBlockTypes = "EncoderOutBlock",
        downsample_type: DownsampleTypes | Sequence[DownsampleTypes] = "Downsample",
        act_fn: ActivationTypes = "silu", 
        norm_type: NormTypes = "group",
        num_groups: int = 8,
        kernel_size: int = 3,
        res_args: dict = {},
        attn_args: dict = {},
        double_z: bool = True,
        out_shortcut: bool = False,
    ):
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
            downsample_type: Type of downsampling block, per level (see
                `chuchichaestli.models.downsampling` for details).
            act_fn: Activation function for the output layers
                (see `chuchichaestli.models.activations` for details).
            norm_type: Normalization type for the output layer.
            num_groups: Number of groups for normalization in the output layer.
            kernel_size: Kernel size for the output convolution.
            res_args: Arguments for residual blocks. Each value is either a
                single value or one per block position, in order of data flow.
            attn_args: Arguments for attention blocks, per block position or per
                block that has attention. `norm_type`, `scales`, `context_args`
                and `local_args` are passed through unchanged.
            double_z: Whether to double the latent space.
            out_shortcut: Whether to use a shortcut for the output block.
        """
        super().__init__()

        block_out_channel_mults = tuple(block_out_channel_mults)
        if len(block_out_channel_mults) < len(down_block_types):
            block_out_channel_mults += (1,) * (
                len(down_block_types) - len(block_out_channel_mults)
            )
        elif len(block_out_channel_mults) > len(down_block_types):
            block_out_channel_mults = block_out_channel_mults[: len(down_block_types)]
        n_mults = len(block_out_channel_mults)
        self.channel_mults = prod(block_out_channel_mults)
        if isinstance(num_layers_per_block, int):
            num_layers_per_block = (num_layers_per_block,) * n_mults
        else:
            num_layers_per_block = tuple(num_layers_per_block)
            if len(num_layers_per_block) < len(down_block_types):
                num_layers_per_block += (num_layers_per_block[-1],) * (
                    len(down_block_types) - len(num_layers_per_block)
                )

        # One sampler entry per level; the last governs channel bookkeeping only
        downsample_types = broadcast(
            downsample_type,
            n_mults,
            "downsample_type",
            None,
            f"[{n_mults} level(s)]",
        )
        downsample_clss = [
            require_cls(name, DOWNSAMPLE_FUNCTIONS, "downsampling type")
            for name in downsample_types
        ]

        # Block positions in order of data flow
        n_pos = n_mults + len(mid_block_types)
        path = f"[{n_mults} level(s), {len(mid_block_types)} mid block(s)]"
        has_attention = [
            block_type in ATTN_BLOCK_MAP
            for block_type in (*down_block_types, *mid_block_types)
        ]
        res_args = broadcast_kwargs(res_args, n_pos, context=path)
        attn_args = broadcast_kwargs(
            attn_args,
            n_pos,
            mask=has_attention,
            opaque=INTRA_BLOCK_ATTN_ARGS,
            context=path,
        )

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
            outs = ins
            if not downsample_clss[i].changes_channels:
                outs = int(ins * block_out_channel_mults[i])
            stage = nn.Sequential()
            for _ in range(num_layers_per_block[i]):
                down_block = BLOCK_MAP[down_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    res_args=res_args[i],
                    attn_args=attn_args[i],
                )
                stage.append(down_block)
                ins = outs
            self.down_blocks.append(stage)

            if i < n_mults - 1:
                if not downsample_clss[i].changes_channels:
                    self.down_blocks.append(downsample_clss[i](dimensions, ins))
                else:
                    self.down_blocks.append(
                        downsample_clss[i](
                            dimensions,
                            ins,
                            outs := int(ins * block_out_channel_mults[i]),
                        )
                    )
                    ins = outs

        self.mid_blocks = nn.ModuleList([])
        for j, mid_block_type in enumerate(mid_block_types):
            mid_block = BLOCK_MAP[mid_block_type](
                dimensions=dimensions,
                channels=outs,
                res_args=res_args[n_mults + j],
                attn_args=attn_args[n_mults + j],
            )
            self.mid_blocks.append(mid_block)

        self.levels = (len(self.down_blocks) + 1) // 2
        self.out_channels = 2 * out_channels if double_z else out_channels
        self.out_block = BLOCK_MAP[out_block_type](
            dimensions=dimensions,
            in_channels=outs,
            out_channels=self.out_channels,
            act_fn=act_fn,
            norm_type=norm_type,
            num_groups=num_groups,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.out_shortcut = out_shortcut
        self.shortcut_groups = outs // self.out_channels

    @property
    def f(self) -> int:
        """Compression factor of the encoder.

        Raises:
            ValueError: If a sampling block does not scale by a constant factor.
        """
        factors = [
            getattr(b, "factor", None) for b in self.down_blocks if hasattr(b, "factor")
        ]
        if any(factor is None for factor in factors):
            raise ValueError(
                "A sampling block pools to a fixed size rather than by a factor,"
                " so the model has no constant scaling factor."
            )
        return prod(factors)

    def forward(self, x):
        """Forward pass."""
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        for block in self.mid_blocks:
            h = block(h)
        if self.out_shortcut:
            shortcut = h.unflatten(1, (-1, self.shortcut_groups)).mean(dim=2)
            h = self.out_block(h) + shortcut
        else:
            h = self.out_block(h)
        return h

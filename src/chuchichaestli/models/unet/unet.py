# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A highly customizable U-Net model implementation."""

import warnings
import torch
from torch import nn

from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.blocks import (
    GaussianNoiseBlock,
    ATTN_BLOCK_MAP,
    BLOCK_MAP,
    CONV_BLOCK_MAP,
    UNetDownBlockTypes,
    UNetMidBlockTypes,
    UNetUpBlockTypes,
)
from chuchichaestli.models.downsampling import (
    DOWNSAMPLE_FUNCTIONS,
    DownsampleTypes,
)
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, require_cls
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.unet.time_embeddings import (
    TimeEmbeddingTypes,
    TIME_EMBEDDING_MAP,
)
from chuchichaestli.models.upsampling import (
    UPSAMPLE_FUNCTIONS,
    UpsampleTypes,
)
from chuchichaestli.utils import broadcast, broadcast_kwargs
from typing import Literal
from collections.abc import Sequence

SkipConnectionTypes = Literal["concat", "avg", "add"]


class UNet(nn.Module):
    """Highly customizable U-Net model implementation.

    The architecture consists of an encoder-decoder structure with skip connections.
    The encoder chains several convolutional (residual) and downsampling blocks.
    Each downsampling block separates the encoder into spatially hierarchical levels.
    The decoder is built symmetrically to the encoder with (residual) transposed
    convolutional and upsampling blocks, each level linked via skip connections
    which ensure spatial information is passed through the network.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 32,
        out_channels: int = 1,
        down_block_types: Sequence[UNetDownBlockTypes] = (
            "DownBlock",
            "DownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
        ),
        mid_block_type: UNetMidBlockTypes = "MidBlock",
        up_block_types: Sequence[UNetUpBlockTypes] = (
            "AttnUpBlock",
            "AttnUpBlock",
            "UpBlock",
            "UpBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 4),
        num_blocks_per_level: int = 1,
        upsample_type: UpsampleTypes | Sequence[UpsampleTypes] = "Upsample",
        downsample_type: DownsampleTypes | Sequence[DownsampleTypes] = "Downsample",
        act_fn: ActivationTypes = "silu",
        norm_type: NormTypes = "group",
        groups: int = 8,
        in_kernel_size: int = 3,
        out_kernel_size: int = 3,
        time_embedding: TimeEmbeddingTypes | bool | None = None,
        time_channels: int = 32,
        t_emb_dim: int = 32,
        t_emb_flip: bool = False,
        t_emb_shift: float = 1.0,
        t_emb_act_fn: ActivationTypes = "silu",
        t_emb_post_act: bool = False,
        t_emb_condition_dim: int | None = None,
        res_act_fn: ActivationTypes | Sequence[ActivationTypes] = "silu",
        res_dropout: float | Sequence[float] = 0.1,
        res_norm_type: NormTypes | Sequence[NormTypes] = "group",
        res_groups: int | Sequence[int] = 32,
        res_kernel_size: int | Sequence[int] = 3,
        attn_head_dim: int | Sequence[int] = 32,
        attn_n_heads: int | Sequence[int] = 1,
        attn_dropout_p: float | Sequence[float] = 0.0,
        attn_norm_type: NormTypes | Sequence[NormTypes] = "group",
        attn_groups: int | Sequence[int] = 32,
        attn_kernel_size: int | Sequence[int] = 1,
        attn_gate_inter_channels: int | Sequence[int] | None = None,
        attn_gate_subsample_factor: int | Sequence[int] = 1,
        attn_gate_out_norm_type: NormTypes
        | None
        | Sequence[NormTypes | None] = "batch",
        skip_connection_action: SkipConnectionTypes
        | None
        | Sequence[SkipConnectionTypes | None] = "concat",
        skip_connection_to_all_blocks: bool | None = None,
        add_noise: Literal["up", "down"] | None = None,
        noise_sigma: float = 0.1,
        noise_detached: bool = True,
    ):
        """Constructor.

        Args:
            dimensions: Number of (spatial) dimensions.
            in_channels: Number of input channels.
            n_channels: Number of channels in the first block.
            out_channels: Number of output channels.
            down_block_types: Types of down blocks as a list, starting at the
                first block in the highest level (in data-flow order).
            mid_block_type: Type of mid block.
            up_block_types: Types of up blocks as a list, starting at the
                first block in the lowest level (in data-flow order).
            block_out_channel_mults: Output channel multipliers for each block.
            num_blocks_per_level: Number of blocks per level
                (blocks are repeated if `>1`).
            upsample_type: Type of upsampling block, per level transition
                (see `chuchichaestli.models.upsampling` for details).
            downsample_type: Type of downsampling block, per level transition
                (see `chuchichaestli.models.downsampling` for details).
            act_fn: Activation function for the output layer
                (see `chuchichaestli.models.activations` for details).
            norm_type: Normalization type for the output layer.
            groups: Number of groups for group normalization in the output layer.
            in_kernel_size: Kernel size for the input convolution.
            out_kernel_size: Kernel size for the output convolution.
            time_embedding: Whether to use a time embedding.
            time_channels: Number of time channels.
            t_emb_dim: The dimension for the deep embedding (takes only effect
                if `time_embedding='DeepSinusoidalTimeEmbedding'`).
            t_emb_flip: Whether to flip the sine to cosine in the time embedding.
            t_emb_shift: The downscale frequency shift for the time embedding.
            t_emb_act_fn: Activation function for the time embedding.
            t_emb_post_act: Whether to use an activation at the end of the time embedding.
            t_emb_condition_dim: The condition dimension for the time embedding.
            res_act_fn: Activation function for the residual blocks, per block
                position (see `chuchichaestli.models.activations` for details).
            res_dropout: Dropout rate for the residual blocks, per block position.
            res_norm_type: Normalization type for the residual blocks, per block
                position (see `chuchichaestli.models.norm` for details).
            res_groups: Number of groups for the residual block normalization
                (if group norm), per block position.
            res_kernel_size: Kernel size for the residual blocks, per block position.
            attn_head_dim: Dimension of the attention heads, per attention block.
            attn_n_heads: Number of attention heads, per attention block.
            attn_dropout_p: Dropout probability of the scaled dot product attention,
                per attention block.
            attn_norm_type: Normalization type for the convolutional attention block,
                per attention block (see `chuchichaestli.models.norm` for details).
            attn_groups: Number of groups for the convolutional attention block
                normalization (if `attn_norm_type` is `"group"`), per attention block.
            attn_kernel_size: Kernel size for the convolutional attention block,
                per attention block.
            attn_gate_inter_channels: Number of intermediate channels for the attention
                gate (if `up_block_types` contains `"AttnGateUpBlock"`), per attention
                block; halves the block's channels by default.
            attn_gate_subsample_factor: Stride at which an attention gate samples the
                skip connection, per attention block (see
                `chuchichaestli.models.attention.AttentionGate`).
            attn_gate_out_norm_type: Normalization after an attention gate's output
                transform. Defaults to `"batch"`, as in the reference.
            skip_connection_action: Action to take for the skip connection, per up
                level. If `None`, no skip connection is used at that level.
            skip_connection_to_all_blocks: If `True`, the U-Net builds skip connections
                to all blocks in a level, otherwise only to the first block in a level.
            add_noise: Add a Gaussian noise regularizer block in the bottleneck (before or after).
                Can be "up" (after the bottleneck) or "down" (before the bottleneck).
            noise_sigma: Std. relative (to the magnitude of the input) for the noise generation.
            noise_detached: If True, the input is detached for the noise generation.
                Note, this should generally be `True`, otherwise the noise is learnable.
        """
        super().__init__()

        self._validate_inputs(
            dimensions,
            down_block_types,
            up_block_types,
            block_out_channel_mults,
            num_blocks_per_level,
        )

        # Cache commonly used values
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        n_mults = len(block_out_channel_mults)
        self.num_blocks_per_level = num_blocks_per_level
        self.skip_connection_to_all_blocks = skip_connection_to_all_blocks

        # Block positions in order of data flow: down levels, mid block, up levels
        n_pos = 2 * n_mults + 1
        path = f"[{n_mults} down level(s), mid block, {n_mults} up level(s)]"
        has_attention = [
            block_type in ATTN_BLOCK_MAP
            for block_type in (*down_block_types, mid_block_type, *up_block_types)
        ]

        # Up-/Downsampling block broadcasting
        n_samplers = n_mults - 1
        smplr_err_ctx = f"[{n_samplers} sampling block(s)]"
        downsample_types = broadcast(
            downsample_type, n_samplers, "downsample_type", None, smplr_err_ctx
        )
        upsample_types = broadcast(
            upsample_type, n_samplers, "upsample_type", None, smplr_err_ctx
        )
        downsample_clss = [
            require_cls(n, DOWNSAMPLE_FUNCTIONS, "sampling type for a U-Net")
            for n in downsample_types
        ]
        upsample_clss = [
            require_cls(n, UPSAMPLE_FUNCTIONS, "sampling type for a U-Net")
            for n in upsample_types
        ]
        down_changes_channels = [cls.changes_channels for cls in downsample_clss]
        up_changes_channels = [cls.changes_channels for cls in upsample_clss]
        mismatched = [
            i
            for i in range(n_samplers)
            if down_changes_channels[i] != up_changes_channels[n_samplers - 1 - i]
        ]
        if mismatched:
            raise ValueError(
                f"The down and up sampling types must agree on which levels change the"
                f" channel count; they differ at level(s) {mismatched}."
            )
        # the last level has no sampler, so its blocks apply the multiplier
        down_changes_channels.append(False)

        skip_actions = broadcast(
            skip_connection_action,
            n_mults,
            "skip_connection_action",
            None,
            f"[{n_mults} up level(s)]",
        )

        # Group normalization configuration
        res_norm_types = broadcast(res_norm_type, n_pos, "res_norm_type", None, path)
        res_groups_per_pos = broadcast(res_groups, n_pos, "res_groups", None, path)
        replaced_groups: list[int | None] = []

        # Pre-compute argument dictionaries, one per block position
        res_args = broadcast_kwargs(
            {
                "res_groups": res_groups_per_pos,
                "res_act_fn": res_act_fn,
                "res_dropout": res_dropout,
                "res_norm_type": res_norm_types,
                "res_kernel_size": res_kernel_size,
            },
            n_pos,
            context=path,
        )

        attn_args = broadcast_kwargs(
            {
                "n_heads": attn_n_heads,
                "head_dim": attn_head_dim,
                "dropout_p": attn_dropout_p,
                "norm_type": attn_norm_type,
                "groups": attn_groups,
                "kernel_size": attn_kernel_size,
                "num_channels_inter": attn_gate_inter_channels,
                "subsample_factor": attn_gate_subsample_factor,
                "out_norm_type": attn_gate_out_norm_type,
            },
            n_pos,
            mask=has_attention,
            context=path,
        )

        # Input layer
        self.conv_in = conv_cls(
            in_channels, n_channels, kernel_size=in_kernel_size, padding="same"
        )

        self.time_emb = (
            TIME_EMBEDDING_MAP[time_embedding](
                num_channels=time_channels,
                embedding_dim=t_emb_dim,
                flip_sin_to_cos=t_emb_flip,
                downscale_freq_shift=t_emb_shift,
                activation=t_emb_act_fn,
                post_activation=t_emb_post_act,
                condition_dim=t_emb_condition_dim,
            )
            if time_embedding
            else None
        )

        # Build encoder
        self.down_blocks = nn.ModuleList([])
        self.skip_sources: list[bool] = []
        ins = n_channels
        for i in range(n_mults):
            outs = ins if down_changes_channels[i] else ins * block_out_channel_mults[i]
            replaced_groups.append(
                self._clamp_groups(res_args[i], res_norm_types[i], min(ins, outs))
            )

            for j in range(num_blocks_per_level):
                down_block = BLOCK_MAP[down_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    time_embedding=self.time_emb is not None,
                    time_channels=time_channels,
                    res_args=res_args[i],
                    attn_args=attn_args[i],
                )
                self.down_blocks.append(down_block)
                # only the last block in a level feeds a skip connection
                self.skip_sources.append(j == num_blocks_per_level - 1)
                ins = outs

            if i < n_mults - 1:
                if down_changes_channels[i]:
                    widened = ins * block_out_channel_mults[i]
                    self.down_blocks.append(
                        downsample_clss[i](dimensions, ins, widened)
                    )
                    ins = outs = widened
                else:
                    self.down_blocks.append(downsample_clss[i](dimensions, ins))
                self.skip_sources.append(False)

        # Build middle block
        replaced_groups.append(
            self._clamp_groups(res_args[n_mults], res_norm_types[n_mults], outs)
        )
        self.mid_block = BLOCK_MAP[mid_block_type](
            dimensions=dimensions,
            channels=outs,
            time_embedding=self.time_emb is not None,
            time_channels=time_channels,
            res_args=res_args[n_mults],
            attn_args=attn_args[n_mults],
        )

        # Build decoder
        self.up_blocks = nn.ModuleList([])
        self.up_samplers: list[bool] = []
        self.up_level_starts: list[bool] = []

        for i, up_block_type in enumerate(up_block_types):
            level = n_mults - 1 - i
            ins = outs
            outs = (
                ins
                if down_changes_channels[level]
                else ins // block_out_channel_mults[level]
            )
            pos = n_mults + 1 + i
            replaced_groups.append(
                self._clamp_groups(res_args[pos], res_norm_types[pos], min(ins, outs))
            )

            for j in range(num_blocks_per_level):
                up_block = BLOCK_MAP[up_block_type](
                    dimensions=dimensions,
                    in_channels=ins if j == 0 else outs,
                    out_channels=outs,
                    time_embedding=self.time_emb is not None,
                    time_channels=time_channels,
                    res_args=res_args[n_mults + 1 + i],
                    attn_args=attn_args[n_mults + 1 + i],
                    # every block in a level is fed that level's skip connection
                    skip_channels=ins,
                    skip_connection_action=(
                        skip_actions[i]
                        if j == 0 or skip_connection_to_all_blocks
                        else None
                    ),
                )
                self.up_blocks.append(up_block)
                # the first block in a level consumes the skip connection
                self.up_samplers.append(False)
                self.up_level_starts.append(j == 0)

            if i < n_mults - 1:
                mirrored = n_mults - 2 - i
                if down_changes_channels[mirrored]:
                    narrowed = outs // block_out_channel_mults[mirrored]
                    self.up_blocks.append(upsample_clss[i](dimensions, outs, narrowed))
                    outs = narrowed
                else:
                    self.up_blocks.append(upsample_clss[i](dimensions, outs))
                self.up_samplers.append(True)
                self.up_level_starts.append(False)

        match add_noise:
            case "up":
                self.up_blocks.insert(
                    0, GaussianNoiseBlock(sigma=noise_sigma, detached=noise_detached)
                )
                self.up_samplers.insert(0, True)
                self.up_level_starts.insert(0, False)
            case "down":
                self.down_blocks.append(
                    GaussianNoiseBlock(sigma=noise_sigma, detached=noise_detached)
                )
                self.skip_sources.append(False)

        indivisible_groups = {g for g in replaced_groups if g is not None}
        if indivisible_groups:
            # one warning for the model, not one per non-divisible block position
            groups_str = ", ".join(str(g) for g in sorted(indivisible_groups))
            warnings.warn(
                f"Number of channels is not divisible by the number of groups"
                f" ({groups_str}) at some block positions."
                f" Setting those to the block's own channel count."
            )
            groups = min(groups, n_channels)

        # Output layer
        self.out_block = CONV_BLOCK_MAP["NormActConvBlock"](
            dimensions=dimensions,
            in_channels=outs,
            out_channels=out_channels,
            act_fn=act_fn,
            norm_type=norm_type,
            num_groups=groups,
            kernel_size=out_kernel_size,
            stride=1,
            padding="same",
        )

    def _clamp_groups(
        self, res_args: dict, norm_type: NormTypes, channels: int
    ) -> int | None:
        """Reduce a group count that does not divide the channels it normalizes.

        Args:
            res_args: Residual block arguments of one position, adjusted in place.
            norm_type: Normalization type at that position.
            channels: Narrowest channel count the position normalizes.

        Returns:
            The group count that had to be replaced, or `None` if it fits.
        """
        groups = res_args["res_groups"]
        if norm_type != "group" or channels % groups == 0:
            return None
        res_args["res_groups"] = channels
        return groups

    def _validate_inputs(
        self,
        dimensions,
        down_block_types,
        up_block_types,
        block_out_channel_mults,
        num_blocks_per_level,
    ):
        """Validate constructor inputs."""
        if dimensions not in DIM_TO_CONV_MAP:
            raise ValueError(
                f"Invalid number of dimensions ({dimensions}). Must be one of {list(DIM_TO_CONV_MAP.keys())}."
            )

        if len(down_block_types) != len(up_block_types):
            raise ValueError("The number of down and up block types must be equal.")

        if len(down_block_types) != len(block_out_channel_mults):
            raise ValueError(
                "The number of down block types and output channel multipliers must be equal."
            )

        if num_blocks_per_level < 1:
            raise ValueError(
                f"Each level needs at least one block;"
                f" got num_blocks_per_level={num_blocks_per_level}."
            )

    def forward(
        self, x: torch.Tensor, t: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            t: Time step, if the U-Net was built with a time embedding.
        """
        t_emb = None
        if t is not None and self.time_emb is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=x.device)
            t_emb = self.time_emb(t.expand(x.shape[0]))

        x = self.conv_in(x)

        hh = []
        for down_block, is_skip_source in zip(
            self.down_blocks, self.skip_sources, strict=True
        ):
            x = down_block(x, t_emb)
            if is_skip_source:
                hh.append(x)

        x = self.mid_block(x, t_emb)

        hs = None
        for up_block, is_sampler, is_level_start in zip(
            self.up_blocks, self.up_samplers, self.up_level_starts, strict=True
        ):
            if is_sampler:
                x = up_block(x, t_emb)
                continue
            if is_level_start:
                hs = hh.pop()
            x = up_block(x, hs, t_emb)
        return self.out_block(x)

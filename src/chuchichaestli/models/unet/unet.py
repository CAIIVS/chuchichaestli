# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A highly customizable U-Net model implementation."""

import warnings
import torch
from torch import nn

from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.blocks import (
    BLOCK_MAP,
    CONV_BLOCK_MAP,
    GaussianNoiseBlock,
    UNetDownBlockTypes,
    UNetMidBlockTypes,
    UNetUpBlockTypes,
)
from chuchichaestli.models.downsampling import (
    DOWNSAMPLE_FUNCTIONS,
    Downsample,
    DownsampleInterpolate,
    DownsampleTypes,
)
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.unet.time_embeddings import (
    SinusoidalTimeEmbedding,
    DeepSinusoidalTimeEmbedding,
)
from chuchichaestli.models.upsampling import (
    UPSAMPLE_FUNCTIONS,
    Upsample,
    UpsampleInterpolate,
    UpsampleTypes,
)
from typing import Literal
from collections.abc import Sequence


TIME_EMBEDDING_MAP = {
    "SinusoidalTimeEmbedding": SinusoidalTimeEmbedding,
    "DeepSinusoidalTimeEmbedding": DeepSinusoidalTimeEmbedding,
    True: SinusoidalTimeEmbedding,
}


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
            "UpBlock",
            "UpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 4),
        num_blocks_per_level: int = 1,
        upsample_type: UpsampleTypes = "Upsample",
        downsample_type: DownsampleTypes = "Downsample",
        act_fn: ActivationTypes = "silu",
        norm_type: NormTypes = "group",
        groups: int = 8,
        in_kernel_size: int = 3,
        out_kernel_size: int = 3,
        time_embedding: Literal[
            "SinusoidalTimeEmbedding", "DeepSinusoidalTimeEmbedding"
        ]
        | bool
        | None = None,
        time_channels: int = 32,
        t_emb_dim: int = 32,
        t_emb_flip: bool = False,
        t_emb_shift: float = 1.0,
        t_emb_act_fn: ActivationTypes = "silu",
        t_emb_post_act: bool = False,
        t_emb_condition_dim: int | None = None,
        res_act_fn: ActivationTypes = "silu",
        res_dropout: float = 0.1,
        res_norm_type: NormTypes = "group",
        res_groups: int = 32,
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: NormTypes = "group",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_gate_inter_channels: int = 32,
        skip_connection_action: Literal["concat", "avg", "add"] | None = "concat",
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
                first block (in the highest level).
            mid_block_type: Type of mid block.
            up_block_types: Types of up blocks as a list, starting with the last
                block (lowest level).
            block_out_channel_mults: Output channel multipliers for each block.
            num_blocks_per_level: Number of blocks per level
                (blocks are repeated if `>1`).
            upsample_type: Type of upsampling block
                (see `chuchichaestli.models.upsampling` for details).
            downsample_type: Type of downsampling block
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
            res_act_fn: Activation function for the residual blocks
                (see `chuchichaestli.models.activations` for details).
            res_dropout: Dropout rate for the residual blocks.
            res_norm_type: Normalization type for the residual block
                (see `chuchichaestli.models.norm` for details).
            res_groups: Number of groups for the residual block normalization (if group norm).
            res_kernel_size: Kernel size for the residual blocks.
            attn_head_dim: Dimension of the attention heads.
            attn_n_heads: Number of attention heads.
            attn_dropout_p: Dropout probability of the scaled dot product attention.
            attn_norm_type: Normalization type for the convolutional attention block
                (see `chuchichaestli.models.norm` for details).
            attn_groups: Number of groups for the convolutional attention block normalization
                (if `attn_norm_type` is `"group"`).
            attn_kernel_size: Kernel size for the convolutional attention block.
            attn_gate_inter_channels: Number of intermediate channels for the attention gate
                (if `up_block_types` contains `"AttnGateUpBlock"`).
            skip_connection_action: Action to take for the skip connection.
                If `None`, no skip connections are used.
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
            dimensions, down_block_types, up_block_types, block_out_channel_mults
        )

        # Cache commonly used values
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        upsample_cls = UPSAMPLE_FUNCTIONS[upsample_type]
        downsample_cls = DOWNSAMPLE_FUNCTIONS[downsample_type]
        n_mults = len(block_out_channel_mults)
        self.num_blocks_per_level = num_blocks_per_level
        self.skip_connection_to_all_blocks = skip_connection_to_all_blocks

        # Group normalization configuration
        if res_norm_type == "group" and n_channels % res_groups != 0:
            warnings.warn(
                f"Number of channels ({n_channels}) is not divisible by the number of groups ({res_groups}). Setting number of groups to n_channels."
            )
            res_groups = n_channels
            groups = min(groups, n_channels)

        # Pre-compute argument dictionaries to avoid repeated dict creation
        res_args = {
            "res_groups": res_groups,
            "res_act_fn": res_act_fn,
            "res_dropout": res_dropout,
            "res_norm_type": res_norm_type,
            "res_kernel_size": res_kernel_size,
        }

        attn_args = {
            "n_heads": attn_n_heads,
            "head_dim": attn_head_dim,
            "dropout_p": attn_dropout_p,
            "norm_type": attn_norm_type,
            "groups": attn_groups,
            "kernel_size": attn_kernel_size,
            "inter_channels": attn_gate_inter_channels,
        }

        # Input layer
        self.conv_in = conv_cls(
            in_channels, n_channels, kernel_size=in_kernel_size, padding="same"
        )

        self.time_channels = time_channels
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
        ins = n_channels
        for i in range(n_mults):
            outs = ins * block_out_channel_mults[i]

            for _ in range(num_blocks_per_level):
                down_block = BLOCK_MAP[down_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    time_embedding=self.time_emb is not None,
                    time_channels=time_channels,
                    res_args=res_args,
                    attn_args=attn_args,
                )
                self.down_blocks.append(down_block)
                ins = outs

            if i < n_mults - 1:
                self.down_blocks.append(downsample_cls(dimensions, ins))

        # Build middle block
        self.mid_block = BLOCK_MAP[mid_block_type](
            dimensions=dimensions,
            channels=outs,
            time_embedding=self.time_emb is not None,
            time_channels=time_channels,
            res_args=res_args,
            attn_args=attn_args,
        )

        # Build decoder
        self.up_blocks = nn.ModuleList([])

        for i in reversed(range(n_mults)):
            ins = outs
            outs = ins // block_out_channel_mults[i]

            for j in range(num_blocks_per_level):
                up_block = BLOCK_MAP[up_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins if j == 0 else outs,
                    out_channels=outs,
                    time_embedding=self.time_emb is not None,
                    time_channels=time_channels,
                    res_args=res_args,
                    attn_args=attn_args,
                    skip_connection_action=(
                        skip_connection_action
                        if j == 0 or skip_connection_to_all_blocks
                        else None
                    ),
                )
                self.up_blocks.append(up_block)

            ins = outs
            if i > 0:
                self.up_blocks.append(upsample_cls(dimensions, outs))

        match add_noise:
            case "up":
                self.up_blocks.insert(
                    0, GaussianNoiseBlock(sigma=noise_sigma, detached=noise_detached)
                )
            case "down":
                self.down_blocks.append(
                    GaussianNoiseBlock(sigma=noise_sigma, detached=noise_detached)
                )

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

    def _validate_inputs(
        self, dimensions, down_block_types, up_block_types, block_out_channel_mults
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

    def forward(
        self, x: torch.Tensor, t: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass - optimized but maintains exact original logic."""
        t_emb = None
        if t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=x.device)
            t = t.expand(x.shape[0])
            t_emb = self.time_emb(t) if self.time_emb is not None else None

        x = self.conv_in(x)

        hh = []
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x, t_emb)
            if isinstance(
                down_block, Downsample | DownsampleInterpolate | GaussianNoiseBlock
            ):
                continue
            # Append skip connection for the last down_block in each layer
            if (i + 1) % self.num_blocks_per_level == 0:
                hh.append(x)

        x = self.mid_block(x, t_emb)

        no_count_block = 0
        for i, up_block in enumerate(self.up_blocks):
            if isinstance(
                up_block, Upsample | UpsampleInterpolate | GaussianNoiseBlock
            ):
                x = up_block(x, t_emb)
                no_count_block += 1
                continue
            # concat skip connection for the first upblock of each layer
            if (i - no_count_block) % self.num_blocks_per_level == 0:
                hs = hh.pop()
                x = up_block(x, hs, t_emb)
            elif self.skip_connection_to_all_blocks:
                hs = hh[-1]
                x = up_block(x, hs, t_emb)
            else:
                x = up_block(x=x, h=None, t=t_emb)
        x = self.out_block(x)
        return x

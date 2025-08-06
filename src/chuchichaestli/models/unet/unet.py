"""UNet model - Conservatively Optimized version.

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

import warnings
import torch
from torch import nn

from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.downsampling import (
    DOWNSAMPLE_FUNCTIONS,
    Downsample,
    DownsampleInterpolate,
)
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import Norm
from chuchichaestli.models.unet.blocks import (
    AttnDownBlock,
    AttnGateUpBlock,
    AttnMidBlock,
    AttnUpBlock,
    ConvAttnDownBlock,
    ConvAttnMidBlock,
    ConvAttnUpBlock,
    DownBlock,
    GaussianNoiseBlock,
    MidBlock,
    UpBlock,
)
from chuchichaestli.models.unet.time_embeddings import (
    SinusoidalTimeEmbedding,
)
from chuchichaestli.models.upsampling import (
    UPSAMPLE_FUNCTIONS,
    Upsample,
    UpsampleInterpolate,
)
from typing import Literal
from collections.abc import Sequence

BLOCK_MAP = {
    "DownBlock": DownBlock,
    "MidBlock": MidBlock,
    "UpBlock": UpBlock,
    "AttnDownBlock": AttnDownBlock,
    "AttnMidBlock": AttnMidBlock,
    "AttnUpBlock": AttnUpBlock,
    "ConvAttnDownBlock": ConvAttnDownBlock,
    "ConvAttnMidBlock": ConvAttnMidBlock,
    "ConvAttnUpBlock": ConvAttnUpBlock,
    "AttnGateUpBlock": AttnGateUpBlock,
}


class UNet(nn.Module):
    """Flexible U-Net model implementation."""

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 32,
        out_channels: int = 1,
        down_block_types: Sequence[
            Literal["DownBlock", "AttnDownBlock", "ConvAttnDownBlock"]
        ] = (
            "DownBlock",
            "DownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
        ),
        mid_block_type: Literal[
            "MidBlock", "AttnMidBlock", "ConvAttnMidBlock"
        ] = "MidBlock",
        up_block_types: Sequence[
            Literal["UpBlock", "AttnUpBlock", "ConvAttnUpBlock", "AttnGateUpBlock"]
        ] = (
            "UpBlock",
            "UpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 4),
        upsample_type: Literal["Upsample", "UpsampleInterpolate"] = "Upsample",
        downsample_type: Literal["Downsample", "DownsampleInterpolate"] = "Downsample",
        num_layers_per_block: int = 1,
        groups: int = 8,
        act: Literal[
            "silu",
            "swish",
            "mish",
            "gelu",
            "relu",
            "prelu",
            "leakyrelu",
            "leakyrelu,0.1",
            "leakyrelu,0.2",
            "softplus",
        ] = "silu",
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
        t_emb_act_fn: Literal[
            "silu",
            "swish",
            "mish",
            "gelu",
            "relu",
            "prelu",
            "leakyrelu",
            "leakyrelu,0.1",
            "leakyrelu,0.2",
            "softplus",
        ] = "silu",
        t_emb_post_act: bool = False,
        t_emb_condition_dim: int | None = None,
        res_groups: int = 32,
        res_act_fn: Literal[
            "silu",
            "swish",
            "mish",
            "gelu",
            "relu",
            "prelu",
            "leakyrelu",
            "leakyrelu,0.1",
            "leakyrelu,0.2",
            "softplus",
        ] = "silu",
        res_dropout: float = 0.1,
        res_norm_type: Literal["group", "instance", "batch", "adabatch"] = "group",
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: Literal["group", "instance", "batch", "adabatch"] = "group",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_gate_inter_channels: int = 32,
        skip_connection_action: Literal["concat", "avg", "add"] = "concat",
        skip_connection_to_all_layers: bool | None = None,
        add_noise: Literal["up", "down"] | None = None,
        noise_sigma: float = 0.1,
        noise_detached: bool = True,
    ):
        """Constructor.

        Args:
            dimensions: Number of (spatial) dimensions.
            in_channels: Number of input channels.
            n_channels: Number of channels in the first layer.
            out_channels: Number of output channels.
            down_block_types: Types of down blocks.
            mid_block_type: Type of mid block.
            up_block_types: Types of up blocks.
            block_out_channel_mults: Output channel multipliers for each block.
            time_embedding: Whether to use a time embedding.
            time_channels: Number of time channels.
            num_layers_per_block: Number of layers per block.
            upsample_type: Type of upsampling block (see `chuchichaestli.models.upsampling` for details).
            downsample_type: Type of downsampling block (see `chuchichaestli.models.downsampling` for details).
            groups: Number of groups for group normalization.
            act: Activation function (see `chuchichaestli.models.activations` for details).
            in_kernel_size: Kernel size for the input convolution.
            out_kernel_size: Kernel size for the output convolution.
            time_embedding: Whether to use a time embedding.
            time_channels: Number of time channels.
            t_emb_dim: The dimension for the deep embedding (takes only
              effect if `time_embedding='DeepSinusoidalTimeEmbedding'`).
            t_emb_flip: Whether to flip the sine to cosine in the time embedding.
            t_emb_shift: The downscale frequency shift for the time embedding.
            t_emb_act_fn: Activation function for the time embedding.
            t_emb_post_act: Whether to use an activation function
              at the end of the time embedding.
            t_emb_condition_dim: The condition dimension for the time embedding.
            res_groups: Number of groups for the residual block normalization (if group norm).
            res_act_fn: Activation function for the residual block
              (see `chuchichaestli.models.activations` for details).
            res_dropout: Dropout rate for the residual block.
            res_norm_type: Normalization type for the residual block
              (see `chuchichaestli.models.norm` for details).
            res_kernel_size: Kernel size for the residual block.
            attn_head_dim: Dimension of the attention head.
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
            skip_connection_to_all_levels: TODO
            add_noise: Add a Gaussian noise regularizer block in the bottleneck (before or after).
              Can be "up" (after the bottlenet) or "down" (before the bottleneck).
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
        self.num_layers_per_block = num_layers_per_block

        # Group normalization optimization
        if res_norm_type == "group" and n_channels % res_groups != 0:
            warnings.warn(
                f"Number of channels ({n_channels}) is not divisible by the number of groups ({res_groups}). Setting number of groups to in_channels."
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

        self.conv_in = conv_cls(
            in_channels, n_channels, kernel_size=in_kernel_size, padding="same"
        )

        self.time_channels = time_channels
        self.time_emb = (
            SinusoidalTimeEmbedding(
                time_channels, flip_sin_to_cos=False, downscale_freq_shift=0.0
            )
            if time_embedding
            else None
        )

        # Build encoder
        self.down_blocks = nn.ModuleList([])
        ins = n_channels
        for i in range(n_mults):
            outs = ins * block_out_channel_mults[i]

            for _ in range(num_layers_per_block):
                down_block = BLOCK_MAP[down_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    time_embedding=time_embedding,
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
            time_embedding=time_embedding,
            time_channels=time_channels,
            res_args=res_args,
            attn_args=attn_args,
        )

        if skip_connection_to_all_layers is None:
            skip_connection_to_all_layers = skip_connection_action == "concat"

        # Build decoder
        self.up_blocks = nn.ModuleList([])

        for i in reversed(range(n_mults)):
            ins = outs
            outs = ins // block_out_channel_mults[i]

            up_block = BLOCK_MAP[up_block_types[i]](
                dimensions=dimensions,
                in_channels=ins,
                out_channels=outs,
                time_embedding=time_embedding,
                time_channels=time_channels,
                res_args=res_args,
                attn_args=attn_args,
                skip_connection_action=skip_connection_action,
            )
            self.up_blocks.append(up_block)

            for _ in range(num_layers_per_block - 1):
                up_block = BLOCK_MAP[up_block_types[i]](
                    dimensions=dimensions,
                    in_channels=outs,
                    out_channels=outs,
                    time_embedding=time_embedding,
                    time_channels=time_channels,
                    res_args=res_args,
                    attn_args=attn_args,
                    skip_connection_action=(
                        skip_connection_action
                        if skip_connection_to_all_layers
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

        # Output layers
        self.norm = Norm(dimensions, res_norm_type, outs, groups)
        self.act = ACTIVATION_FUNCTIONS[act]()
        self.conv_out = conv_cls(
            outs, out_channels, kernel_size=out_kernel_size, padding="same"
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
            if (i + 1) % self.num_layers_per_block == 0:
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
            if (i - no_count_block) % self.num_layers_per_block == 0:
                hs = hh.pop()
                x = up_block(x, hs, t_emb)
            else:
                x = up_block(x=x, h=None, t=t_emb)
        x = self.conv_out(self.act(self.norm(x)))
        return x

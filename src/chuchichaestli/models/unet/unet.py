"""UNet model.

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
from chuchichaestli.models.downsampling import Downsample
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.unet.blocks import (
    AttnDownBlock,
    AttnGateUpBlock,
    AttnMidBlock,
    AttnUpBlock,
    ConvAttnDownBlock,
    ConvAttnMidBlock,
    ConvAttnUpBlock,
    DownBlock,
    MidBlock,
    UpBlock,
    GaussianNoiseBlock,
)
from chuchichaestli.models.unet.time_embeddings import (
    SinusoidalTimeEmbedding,
)
from chuchichaestli.models.upsampling import Upsample
from chuchichaestli.models.norm import Norm


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
    """UNet model implementation."""

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 32,
        out_channels: int = 1,
        down_block_types: tuple[str, ...] = (
            "DownBlock",
            "DownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
        ),
        mid_block_type: str = "MidBlock",
        up_block_types: tuple[str, ...] = (
            "UpBlock",
            "UpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
        ),
        block_out_channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        time_embedding: bool = True,
        time_channels: int = 32,
        num_layers_per_block: int = 1,
        groups: int = 8,
        act: str = "silu",
        in_kernel_size: int = 3,
        out_kernel_size: int = 3,
        res_groups: int = 32,
        res_act_fn: str = "silu",
        res_dropout: float = 0.1,
        res_norm_type: str = "group",
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: str = "group",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_gate_inter_channels: int = 32,
        skip_connection_action: str = "concat",
        skip_connection_between_levels: bool | None = None,
        add_noise: str | None = None,
        noise_sigma: float = 0.1,
        noise_detached: bool = True,
    ):
        """UNet model implementation.

        Args:
            dimensions: Number of dimensions.
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
            groups: Number of groups for group normalization.
            act: Activation function.
            in_kernel_size: Kernel size for the input convolution.
            out_kernel_size: Kernel size for the output convolution.
            res_groups: Number of groups for the residual block normalization (if group norm).
            res_act_fn: Activation function for the residual block.
            res_dropout: Dropout rate for the residual block.
            res_norm_type: Normalization type for the residual block.
            res_kernel_size: Kernel size for the residual block.
            attn_head_dim: Dimension of the attention head.
            attn_n_heads: Number of attention heads.
            attn_dropout_p: Dropout probability of the scaled dot product attention.
            attn_norm_type: Normalization type for the convolutional attention block.
            attn_groups: Number of groups for the convolutional attention block normalization (if group norm).
            attn_kernel_size: Kernel size for the convolutional attention block.
            attn_gate_inter_channels: Number of intermediate channels for the attention gate.
            skip_connection_action: Action to take for the skip connection. Can be "concat", "avg", "add", or None (= do not use skip connections).
            skip_connection_between_levels: Whether to use skip connections between levels (i.e. when channels are not equal). Default is True for "concat" and False for "avg" and "add".
            add_noise: Add a Gaussian noise regularizer block in the bottleneck. Can be "up" (after the bottlenet) or "down" (before the bottleneck).
            noise_sigma: Relative (to the magnitude of the input) standard deviation for the noise generation.
            noise_detached: If True, the input is detached for the noise generation. Note, this should generally be set to True, otherwise the noise is treated as learnable parameter.
        """
        super().__init__()

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

        conv_cls = DIM_TO_CONV_MAP[dimensions]

        if res_norm_type == "group" and n_channels % res_groups != 0:
            warnings.warn(
                f"Number of channels ({n_channels}) is not divisible by the number of groups ({res_groups}). Setting number of groups to in_channels."
            )
            res_groups = n_channels
            groups = min(groups, n_channels)

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

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        n_mults = len(block_out_channel_mults)

        outs = ins = n_channels
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
                self.down_blocks.append(Downsample(dimensions, ins))

        self.mid_block = BLOCK_MAP[mid_block_type](
            dimensions=dimensions,
            channels=outs,
            time_embedding=time_embedding,
            time_channels=time_channels,
            res_args=res_args,
            attn_args=attn_args,
        )

        if skip_connection_between_levels is None:
            skip_connection_between_levels = skip_connection_action == "concat"

        for i in reversed(range(n_mults)):
            outs = ins
            for _ in range(num_layers_per_block):
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

            outs = ins // block_out_channel_mults[i]
            up_block = BLOCK_MAP[up_block_types[i]](
                dimensions=dimensions,
                in_channels=ins,
                out_channels=outs,
                time_embedding=time_embedding,
                time_channels=time_channels,
                res_args=res_args,
                attn_args=attn_args,
                skip_connection_action=skip_connection_action
                if skip_connection_between_levels
                else None,
            )
            self.up_blocks.append(up_block)
            ins = outs
            if i > 0:
                self.up_blocks.append(Upsample(dimensions, outs))

        match add_noise:
            case "up":
                self.up_blocks.insert(
                    0,
                    GaussianNoiseBlock(sigma=noise_sigma, detached=noise_detached)
                )
            case "down":
                self.down_blocks.append(
                    GaussianNoiseBlock(sigma=noise_sigma, detached=noise_detached)
                )

        self.norm = Norm(dimensions, res_norm_type, outs, groups)
        self.act = ACTIVATION_FUNCTIONS[act]()
        self.conv_out = conv_cls(
            outs, out_channels, kernel_size=out_kernel_size, padding="same"
        )

    def forward(
        self, x: torch.Tensor, t: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the UNet model."""
        if t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=x.device)
            t = t.expand(x.shape[0])

            t = self.time_emb(t)
        x = self.conv_in(x)

        hh = [x]

        for down_block in self.down_blocks:
            x = down_block(x, t)
            if isinstance(down_block, GaussianNoiseBlock):
                continue
            hh.append(x)

        x = self.mid_block(x, t)

        for up_block in self.up_blocks:
            if isinstance(up_block, Upsample | GaussianNoiseBlock):
                x = up_block(x, t)
                continue
            hs = hh.pop()
            x = up_block(x, hs, t)

        x = self.conv_out(self.act(self.norm(x)))
        return x

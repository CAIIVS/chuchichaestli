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
    DownBlock,
    MidBlock,
    UpBlock,
)
from chuchichaestli.models.unet.time_embeddings import (
    SinusoidalTimeEmbedding,
)
from chuchichaestli.models.upsampling import Upsample
from chuchichaestli.models.resnet import Norm


BLOCK_MAP = {
    "DownBlock": DownBlock,
    "MidBlock": MidBlock,
    "UpBlock": UpBlock,
    "AttnDownBlock": AttnDownBlock,
    "AttnMidBlock": AttnMidBlock,
    "AttnUpBlock": AttnUpBlock,
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
        attn_gate_inter_channels: int = 32,
    ):
        """UNet model implementation."""
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
            )
            self.up_blocks.append(up_block)
            ins = outs
            if i > 0:
                self.up_blocks.append(Upsample(dimensions, outs))

        self.norm = Norm(dimensions, res_norm_type, outs, groups)
        self.act = ACTIVATION_FUNCTIONS[act]()
        self.conv_out = conv_cls(
            outs, out_channels, kernel_size=out_kernel_size, padding="same"
        )

    def forward(
        self, x: torch.Tensor, t: int | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the UNet model."""
        if t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor([t], dtype=torch.long, device=x.device)
            t *= torch.ones(x.shape[0], dtype=t.dtype, device=t.device)

            t = self.time_emb(t)
        x = self.conv_in(x)

        hh = [x]

        for down_block in self.down_blocks:
            x = down_block(x, t)
            hh.append(x)

        x = self.mid_block(x, t)

        for up_block in self.up_blocks:
            if isinstance(up_block, Upsample):
                x = up_block(x, t)
                continue
            hs = hh.pop()
            x = up_block(x, hs, t)

        x = self.conv_out(self.act(self.norm(x)))
        return x

"""UNet model implementation.

Copyright 2024 The HuggingFace Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Additional modifications made by the Intelligent Vision Systems Group at ZHAW under the
GNU General Public License v3.0 which extends the conditions of the License for further
redistribution and use. See the GPLv3 license at

    http://www.gnu.org/licenses/gpl-3.0.html

This file is part of Chuchichaestli and has been modified for use in this project.
"""

import torch
from torch import nn

from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.downsampling import Downsample
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.unet.blocks import (
    DownBlock,
    MidBlock,
    UpBlock,
)
from chuchichaestli.models.unet.time_embeddings import (
    SinusoidalTimeEmbedding,
)
from chuchichaestli.models.upsampling import Upsample

BLOCK_MAP = {
    "DownBlock": DownBlock,
    "MidBlock": MidBlock,
    "UpBlock": UpBlock,
    "AttnDownBlock": DownBlock,
    "AttnUpBlock": UpBlock,
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
        res_groups: int = 32,
        res_act_fn: str = "silu",
        res_dropout: float = 0.1,
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

        res_args = {
            "res_groups": res_groups,
            "res_act_fn": res_act_fn,
            "res_dropout": res_dropout,
        }

        self.conv_in = conv_cls(in_channels, n_channels, kernel_size=3, padding=1)

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
            )
            self.up_blocks.append(up_block)
            ins = outs
            if i > 0:
                self.up_blocks.append(Upsample(dimensions, outs))

        self.norm = nn.GroupNorm(groups, outs)
        self.act = ACTIVATION_FUNCTIONS[act]()
        self.conv_out = conv_cls(outs, out_channels, kernel_size=3, padding=1)

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
            x = up_block(torch.cat([x, hs], dim=1), t)

        x = self.conv_out(self.act(self.norm(x)))
        return x

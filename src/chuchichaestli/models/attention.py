"""Attention layers.

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
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention block implementation."""

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        n_groups: int = 32,
    ):
        """Attention block implementation."""
        super().__init__()

        if head_dim is None:
            head_dim = n_channels // n_heads

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.proj = nn.Linear(n_channels, n_heads * head_dim * 3)  # q, k, v
        self.output = nn.Linear(n_heads * head_dim, n_channels)

        self.scale = head_dim**-0.5
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention block."""
        batch_size = x.size(0)
        n_channels = x.size(1)
        x = x.view(batch_size, n_channels, -1).moveaxis(1, -1)

        qkv = self.proj(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", attn, v)
        out = out.moveaxis(-1, 1).reshape(batch_size, n_channels, *x.shape[1:])
        out = self.output(out)

        out += x

        return out

"""Self-attention module.

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Attention block implementation."""

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int | None = None,
        dropout_p: float = 0.0,
        **kwargs,
    ):
        """Attention block implementation."""
        super().__init__()

        if head_dim is None:
            head_dim = n_channels // n_heads

        self.proj_in = nn.Linear(n_channels, n_heads * head_dim * 3)  # q, k, v
        self.proj_out = nn.Linear(n_heads * head_dim, n_channels)

        self.scale = head_dim**-0.5
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, _h: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention block."""
        in_shape = x.shape
        x = x.view(in_shape[0], in_shape[1], -1).permute(0, 2, 1)

        qkv = self.proj_in(x).view(in_shape[0], -1, self.n_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.dropout_p
        )
        out = out.reshape(in_shape[0], -1, self.n_heads * self.head_dim)
        out = self.proj_out(out).permute(0, 2, 1).reshape(in_shape)

        return out

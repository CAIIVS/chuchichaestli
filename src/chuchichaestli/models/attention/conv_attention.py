"""Conv-attention module.

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
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import Norm


class ConvAttention(nn.Module):
    """Convolutional attention block implementation.

    Uses convolutions to compute query, key and value matrices.
    """

    def __init__(
        self,
        dimensions: int,
        n_channels: int,
        norm_type: str = "group",
        groups: int = 32,
        kernel_size: int = 1,
        dropout_p: float = 0.0,
        **kwargs,
    ):
        """Convolutional attention block implementation."""
        super().__init__()
        self.norm = Norm(dimensions, norm_type, n_channels, groups)
        self.dropout_p = dropout_p
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.q = conv_cls(n_channels, n_channels, kernel_size=kernel_size)
        self.k = conv_cls(n_channels, n_channels, kernel_size=kernel_size)
        self.v = conv_cls(n_channels, n_channels, kernel_size=kernel_size)
        self.proj_out = conv_cls(n_channels, n_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass of the convolutional attention block."""
        B, C = x.shape[:2]
        h = x

        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = q.view(B, 1, C, -1).permute(0, 1, 3, 2).contiguous()
        k = k.view(B, 1, C, -1).permute(0, 1, 3, 2).contiguous()
        v = v.view(B, 1, C, -1).permute(0, 1, 3, 2).contiguous()
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        return h.permute(0, 3, 1, 2).reshape(x.shape)

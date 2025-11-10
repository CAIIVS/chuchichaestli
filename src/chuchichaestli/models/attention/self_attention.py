# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-attention module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chuchichaestli.models.norm import Norm, NormTypes


class SelfAttention(nn.Module):
    """Attention block implementation."""

    def __init__(
        self,
        dimensions: int,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int | None = None,
        norm_type: NormTypes = "group",
        groups: int = 32,
        dropout_p: float = 0.0,
        **kwargs,
    ):
        """Attention block implementation."""
        super().__init__()

        self.norm = (
            Norm(dimensions, norm_type, n_channels, groups)
            if norm_type is not None
            else None
        )

        if head_dim is None:
            head_dim = n_channels // n_heads

        self.proj_in = nn.Linear(n_channels, n_heads * head_dim * 3)  # q, k, v
        self.proj_out = nn.Linear(n_heads * head_dim, n_channels)

        self.scale = head_dim**-0.5
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass of the attention block."""
        in_shape = x.shape
        x = self.norm(x) if self.norm is not None else x
        x = x.view(in_shape[0], in_shape[1], -1).permute(0, 2, 1)

        qkv = self.proj_in(x).view(in_shape[0], -1, self.n_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.dropout_p
        )
        out = out.reshape(in_shape[0], -1, self.n_heads * self.head_dim)
        out = self.proj_out(out).permute(0, 2, 1).reshape(in_shape)

        return out

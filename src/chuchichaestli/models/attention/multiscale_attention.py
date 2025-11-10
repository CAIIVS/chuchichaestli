# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Conv-attention module."""

from math import gcd
import torch
import torch.nn as nn
import torch.nn.functional as F
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS, ActivationTypes
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import Norm, NormTypes
from collections.abc import Sequence


__all__ = ["LiteMultiscaleAttention"]


def _conv_layer(
    dimensions: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int | str = "same",
    bias: bool = False,
    act_fn: ActivationTypes | None = None,
    norm_type: NormTypes | None = None,
    num_groups: int | None = None,
):
    """Helper function to create a convoltional layer (optionally including activation and normalization)."""
    # Due to circular import issues this cannot be imported from `chuchichaestli.models.blocks`
    # TODO: migrate blocks and use base class to fix this
    conv_cls = DIM_TO_CONV_MAP[dimensions]
    block = nn.Sequential(
        conv_cls(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )
    if act_fn is not None:
        block.append(ACTIVATION_FUNCTIONS[act_fn]())
    if norm_type is not None:
        if norm_type == "group" and (
            in_channels % num_groups != 0 or in_channels < num_groups
        ):
            if in_channels % 2 == 0:
                num_groups = in_channels // 2
            else:
                num_groups = gcd(in_channels, in_channels // 3)
        block.append(
            Norm(
                dimensions,
                norm_type,
                out_channels,
                num_groups,
            )
        )
    return block


class LiteMultiscaleAttention(nn.Module):
    """Lightweight multi-scale attention block implementation.

    Uses convolutions to compute query, key and value matrices.
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        n_heads: int | None = None,
        heads_mult: float = 1,
        head_dim: int = 16,
        scales: Sequence[int] = (5,),
        act_fn: ActivationTypes | Sequence[ActivationTypes | None] | None = None,
        norm_type: NormTypes | Sequence[NormTypes | None] | None = (None, "batch"),
        groups: int | Sequence[int] = 16,
        kernel_size: int = 1,
        bias: bool | Sequence[bool] = False,
        dropout_p: float = 0.0,
        eps: float = 1e-15,
        **kwargs,
    ):
        """Initialize lightweight multi-scale attention block."""
        super().__init__()
        n_heads = (
            int(in_channels // head_dim * heads_mult) if n_heads is None else n_heads
        )
        self.dim = head_dim
        self.total_dim = n_heads * head_dim
        self.eps = eps
        if isinstance(norm_type, str) or norm_type is None:
            norm_type = (norm_type, norm_type)
        if isinstance(groups, int):
            groups = (groups, groups)
        if isinstance(act_fn, str) or act_fn is None:
            act_fn = (act_fn, act_fn)
        if isinstance(bias, bool):
            bias = (bias, bias)
        self.attn_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.qkv = _conv_layer(
            dimensions,
            in_channels,
            self.total_dim * 3,
            act_fn=act_fn[0],
            norm_type=norm_type[0],
            num_groups=groups[0],
            kernel_size=1,
            stride=1,
            padding="same",
            bias=bias[0],
        )
        self.proj_out = _conv_layer(
            dimensions,
            self.total_dim * (1 + len(scales)),
            out_channels,
            act_fn=act_fn[1],
            norm_type=norm_type[1],
            num_groups=groups[1],
            kernel_size=1,
            stride=1,
            padding="same",
            bias=bias[1],
        )
        self.attn_act = ACTIVATION_FUNCTIONS["relu"]()
        self.scale_aggregation = nn.ModuleList(
            [
                nn.Sequential(
                    conv_cls(
                        self.total_dim * 3,
                        self.total_dim * 3,
                        scale,
                        padding="same",
                        groups=self.total_dim * 3,
                        bias=bias[0],
                    ),
                    conv_cls(
                        self.total_dim * 3,
                        self.total_dim * 3,
                        1,
                        padding="same",
                        groups=3 * self.dim,
                        bias=bias[0],
                    ),
                )
                for scale in scales
            ]
        )

    def _relu_lin_attn(self, qkv: torch.Tensor) -> torch.Tensor:
        """Lightweight linear attention with activated query and key."""
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        B = qkv.shape[0]
        spatial_dims = qkv.shape[2:]
        spatial_size = spatial_dims.numel()
        qkv = qkv.reshape(B, -1, 3 * self.dim, spatial_size)
        q, k, v = qkv.chunk(chunks=3, dim=2)
        q = self.attn_act(q)
        k = self.attn_act(k)

        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        if self.training and self.attn_dropout is not None:
            out = self.attn_dropout(out)

        out = torch.reshape(out, (B, -1, *spatial_dims))
        return out

    def _relu_quad_attn(self, qkv: torch.Tensor) -> torch.Tensor:
        """Lightweight quadratic attention with activated query and key."""
        B = qkv.shape[0]
        spatial_dims = qkv.shape[2:]
        spatial_size = spatial_dims.numel()
        qkv = qkv.reshape(B, -1, 3 * self.dim, spatial_size)
        q, k, v = qkv.chunk(chunks=3, dim=2)
        q = self.attn_act(q)
        k = self.attn_act(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)
        dtype = att_map.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)

        if self.training and self.attn_dropout is not None:
            att_map = self.attn_dropout(att_map)

        att_map = att_map.to(dtype)
        out = torch.matmul(v, att_map)
        out = torch.reshape(out, (B, -1, *spatial_dims))
        return out

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass of the multi-scale attention block."""
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for agg in self.scale_aggregation:
            multi_scale_qkv.append(agg(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)
        spatial_size = qkv.shape[2:].numel()
        if spatial_size > self.dim:
            out = self._relu_lin_attn(qkv).to(qkv.dtype)
        else:
            out = self._relu_quad_attn(qkv)
        out = self.proj_out(out)
        return out

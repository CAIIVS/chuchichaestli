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


__all__ = ["MultiscaleLinearAttention"]


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
            out_channels % num_groups != 0 or out_channels < num_groups
        ):
            if out_channels % 2 == 0:
                num_groups = out_channels // 2
            else:
                num_groups = gcd(out_channels, out_channels // 3)
        block.append(
            Norm(
                dimensions,
                norm_type,
                out_channels,
                num_groups,
            )
        )
    return block


class MultiscaleLinearAttention(nn.Module):
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
        attn_act_fn: ActivationTypes = "relu",
        act_fn: ActivationTypes | Sequence[ActivationTypes | None] | None = None,
        norm_type: NormTypes | Sequence[NormTypes | None] | None = (None, "batch"),
        num_groups: int | Sequence[int] = 16,
        kernel_size: int = 1,
        bias: bool | Sequence[bool] = False,
        dropout_p: float = 0.0,
        eps: float = 1e-15,
        **kwargs,
    ):
        """Initialize the multi-scale linear attention block.

        Args:
            dimensions: Number of spatial dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_heads: Number of attention heads.
            heads_mult: Head multiplicity of input channels per head dimension (if `n_heads` is `None`).
            head_dim: Dimensionality of an attention head.
            scales: Convolutional scales for aggregation layers.
            attn_act_fn: Activation function for normalizing query and key.
            act_fn: Activation function(s).
            norm_type: Normalization type(s) for the convolutional layers.
            num_groups: Number of groups for normalization (if `'group'` in `norm_type`).
            kernel_size: Kernel size for the convolutional layers.
            bias: Whether to use bias(es) for the convolutional layers.
            dropout_p: Dropout probability of the block.
            eps: Numerical stability constant.
            kwargs: Additional keyword arguments (only for compatibility).
        """
        super().__init__()
        n_heads = (
            int(in_channels // head_dim * heads_mult) if n_heads is None else n_heads
        )
        self.n_heads = n_heads
        self.dim = head_dim
        self.total_dim = n_heads * head_dim
        self.eps = eps
        if isinstance(norm_type, str) or norm_type is None:
            norm_type = (norm_type, norm_type)
        if isinstance(num_groups, int):
            num_groups = (num_groups, num_groups)
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
            num_groups=num_groups[0],
            kernel_size=kernel_size,
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
            num_groups=num_groups[1],
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=bias[1],
        )
        self.attn_act = ACTIVATION_FUNCTIONS[attn_act_fn]()
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
                        groups=3 * self.n_heads,
                        bias=bias[0],
                    ),
                )
                for scale in scales
            ]
        )

    def _relu_lin_attn(self, qkv: torch.Tensor) -> torch.Tensor:
        """Lightweight linear attention with activated query and key."""
        # The denominator accumulates over every token, so the reduction runs in
        # float32 outside of autocast no matter what dtype the block is called in.
        with torch.autocast(device_type=qkv.device.type, enabled=False):
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
            out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

            if self.training and self.attn_dropout is not None:
                out = self.attn_dropout(out)

            out = torch.reshape(out, (B, -1, *spatial_dims))
        return out

    def _relu_quad_attn(self, qkv: torch.Tensor) -> torch.Tensor:
        """Lightweight quadratic attention with activated query and key."""
        with torch.autocast(device_type=qkv.device.type, enabled=False):
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

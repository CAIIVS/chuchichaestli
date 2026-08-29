# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalization modules for neural networks."""

import torch
from torch import nn
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS, ActivationTypes
from typing import Literal


__all__ = ["Norm", "AdaNorm"]


NormTypes = Literal["group", "instance", "batch", "adabatch", "rms", "layer"]


class Norm(nn.Module):
    """Normalization layer implementation."""

    def __init__(
        self,
        dimensions: int,
        norm_type: NormTypes,
        channels: int,
        num_groups: int,
        affine: bool | None = None,
        **kwargs,
    ):
        """Initialize the normalization layer.

        Args:
            dimensions: Number of (spatial) dimensions.
            norm_type: Type of normalization.
            channels: Number of channels to normalize.
            num_groups: Number of groups, if `norm_type == 'group'`.
            affine: Whether the layer learns its own scale and shift.
                If `None`, the wrapped layer keeps its own default.
            kwargs: Additional keyword arguments for the wrapped layer.
        """
        super().__init__()
        self.ntype = norm_type
        self.norm: nn.Module
        if affine is not None:
            key = "elementwise_affine" if norm_type in ("rms", "layer") else "affine"
            kwargs[key] = affine
        match norm_type:
            case "group":
                self.norm = nn.GroupNorm(num_groups, channels, **kwargs)
            case "instance" if dimensions == 1:
                self.norm = nn.InstanceNorm1d(channels, **kwargs)
            case "instance" if dimensions == 2:
                self.norm = nn.InstanceNorm2d(channels, **kwargs)
            case "instance" if dimensions == 3:
                self.norm = nn.InstanceNorm3d(channels, **kwargs)
            case "batch" if dimensions == 1:
                self.norm = nn.BatchNorm1d(channels, **kwargs)
            case "batch" if dimensions == 2:
                self.norm = nn.BatchNorm2d(channels, **kwargs)
            case "batch" if dimensions == 3:
                self.norm = nn.BatchNorm3d(channels, **kwargs)
            case "rms":
                self.norm = nn.RMSNorm(channels, **kwargs)
            case "layer":
                self.norm = nn.LayerNorm(channels, **kwargs)
            case "adabatch":
                self.norm = AdaptiveBatchNorm(dimensions, channels, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass through the normalization layer."""
        if self.ntype in ("rms", "layer"):
            return self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return self.norm(x)


class AdaNorm(nn.Module):
    """Normalization whose scale and shift are generated from an embedding.

    This is AdaGN for `norm_type='group'` and AdaLN for `'layer'`/`'rms'`, but
    any of `NormTypes` can be modulated:
    The normalization drops its own affine parameters; a linear projection of
    the embedding supplies them instead, one pair per sample:

        AdaNorm(x, emb) = (1 + gamma(emb)) * norm(x) + beta(emb)

    """

    def __init__(
        self,
        dimensions: int,
        norm_type: NormTypes,
        channels: int,
        num_groups: int,
        emb_channels: int,
        act_fn: ActivationTypes = "silu",
        zero_init: bool = True,
        **kwargs,
    ):
        """Initialize the adaptive normalization layer.

        Args:
            dimensions: Number of (spatial) dimensions.
            norm_type: Type of the normalization the embedding modulates.
            channels: Number of channels to normalize.
            num_groups: Number of groups, if `norm_type == 'group'`.
            emb_channels: Number of channels of the modulating embedding.
            act_fn: Activation applied to the embedding before the projection.
            zero_init: If `True`, the projection is zero-initialized, so the
                layer starts out as the plain normalization.
            kwargs: Additional keyword arguments for the wrapped normalization.
        """
        super().__init__()
        self.dimensions = dimensions
        self.norm = Norm(
            dimensions, norm_type, channels, num_groups, affine=False, **kwargs
        )
        self.act = ACTIVATION_FUNCTIONS[act_fn]()
        self.proj = nn.Linear(emb_channels, 2 * channels)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adaptive normalization layer.

        Args:
            x: Input tensor.
            emb: Embedding generating the scale and shift, one row per sample.
        """
        scale, shift = self.proj(self.act(emb)).chunk(2, dim=-1)
        idx = (slice(None), slice(None)) + (None,) * self.dimensions
        return self.norm(x) * (1 + scale[idx]) + shift[idx]


class AdaptiveBatchNorm(nn.Module):
    """Adaptive BN implementation with two additional parameters.

    A batch norm scale `b` and a skip path `a` around the normalization:
        AdaptiveBatchNorm(x) = a * x + b * BatchNorm(x)
    """

    def __init__(
        self,
        dimensions: int,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """Constructor.

        Args:
            dimensions: Number of (spatial) dimensions.
            channels: Number of channels to normalize.
            eps: Offset of the batch normalization denominator.
            momentum: Momentum of the batch normalization running statistics.
            affine: Whether the batch normalization learns its own scale and shift.
        """
        super().__init__()
        self.bn = Norm(
            dimensions,
            "batch",
            channels,
            0,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )
        # start out as plain batch normalization; training mixes in the input
        self.a = nn.Parameter(torch.zeros(1, 1, *((1,) * dimensions)))
        self.b = nn.Parameter(torch.ones(1, 1, *((1,) * dimensions)))

    def forward(self, x):
        """Adaptive BN with two additional parameters `a` and `b`.

        Args:
            x: Input tensor.
        """
        return self.a * x + self.b * self.bn(x)

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Model building blocks for CNNs, U-Nets, autoencoders, and more."""

import torch
from torch import nn
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS, ActivationTypes
from chuchichaestli.models.attention import (
    ATTENTION_MAP,
    AttentionTypes,
    AttentionDownTypes,
)
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import Norm, NormTypes
from chuchichaestli.utils import partialclass, alias_kwargs
from math import gcd
from collections.abc import Callable
from typing import Literal


__all__ = [
    # map dictionaries
    "BLOCK_MAP",
    "CONV_BLOCK_MAP",
    "RESIDUAL_BLOCK_MAP",
    # U-Net block
    "DownBlock",
    "MidBlock",
    "UpBlock",
    "AttnDownBlock",
    "AttnMidBlock",
    "AttnUpBlock",
    "AttnGateUpBlock",
    "ConvAttnDownBlock",
    "ConvAttnMidBlock",
    "ConvAttnUpBlock",
    # Autoencoder blocks
    "AutoencoderDownBlock",
    "AutoencoderMidBlock",
    "AutoencoderUpBlock",
    "AttnAutoencoderDownBlock",
    "AttnAutoencoderMidBlock",
    "AttnAutoencoderUpBlock",
    "ConvAttnAutoencoderDownBlock",
    "ConvAttnAutoencoderMidBlock",
    "ConvAttnAutoencoderUpBlock",
    "EncoderOutBlock",
    "VAEEncoderOutBlock",
    "DecoderInBlock",
    "VAEDecoderInBlock",
    # residual blocks
    "ResidualBlock",
    "ResidualBottleneck",
    # convolutional blocks
    "ConvDownBlock",
    "ConvDownsampleBlock",
    "AttnConvDownBlock",
    "AttnConvDownsampleBlock",
    "ConvBlock",
    "AttnConvBlock",
    "NormConvBlock",
    "NormAttnConvBlock",
    "NormConvDownBlock",
    "NormConvDownsampleBlock",
    "NormAttnConvDownBlock",
    "NormAttnConvDownsampleBlock",
    "ActConvBlock",
    "ActAttnConvBlock",
    "ActConvDownBlock",
    "ActConvDownsampleBlock",
    "ActAttnConvDownBlock",
    "ActAttnConvDownsampleBlock",
    "NormActConvBlock",
    "NormActAttnConvBlock",
    "NormActConvDownBlock",
    "NormActConvDownsampleBlock",
    "NormActAttnConvDownBlock",
    "NormActAttnConvDownsampleBlock",
    # other blocks
    "GaussianNoiseBlock",
]


class GaussianNoiseBlock(nn.Module):
    """Gaussian noise regularization block."""

    def __init__(
        self,
        sigma: float = 0.1,
        mu: float = 0.0,
        detached: bool = True,
        device: torch.device | str | None = None,
    ):
        """Constructor.

        Args:
            sigma: Relative (to the magnitude of the input) standard deviation for noise generation.
            mu: Mean for the noise generation.
            detached: If `True`, the input is detached for the noise generation.
            device: Compute device where to pass the noise.

        Note: If detached=False, the network sees the noise as a trainable parameter
            (no reparametrization trick) and introduce a bias to generate vectors closer
            to the noise level.
        """
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.detached = detached
        self.noise = torch.tensor(mu)

    def forward(
        self, x: torch.Tensor, *args, noise_at_inference: bool = False
    ) -> torch.Tensor:
        """Forward pass using the reparametrization trick."""
        if (self.training or noise_at_inference) and self.sigma != 0:
            scale = self.sigma * x.detach() if self.detached else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class BaseConvBlock(nn.Module):
    """Convolutional block with optional normalization and non-linearity.

    Includes (in following order, unless `not norm_first` or `act_last`):
        - normalization (optional; default: `None`)
        - activation (optional; default: `None`)
        - dropout (optional)
        - attention (optional)
        - convolution (4x4, stride 2)
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        act: bool = True,
        act_fn: ActivationTypes | None = None,
        act_last: bool = False,
        norm: bool = True,
        norm_type: NormTypes | None = None,
        norm_first: bool = True,
        num_groups: int = 16,
        dropout: bool = True,
        dropout_p: float | None = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        double_conv: bool = False,
        attention: AttentionDownTypes | None = None,
        attn_args: dict = {},
        **kwargs,
    ):
        """Initialize a convolutional block with various components.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            act: Use an activation function in the block.
            act_fn: Activation function.
            act_last: If `True`, activations is used last.
            norm: Use normalization in the block.
            norm_type: Normalization type for the conv blocks.
            norm_first: Whether to use normalization before the convolution.
            dropout: Use dropout in the block.
            dropout_p: Dropout probability of the conv blocks.
            num_groups: Number of groups for the conv block normalization (if norm_type == 'group').
            kernel_size: Kernel size for the conv block.
            stride: Stride for the conv block.
            padding: Padding size for the conv block.
            double_conv: Whether to use two convolutional layers.
            attention: Attention descriptor; if None or unknown, no attention is used;
              one of ("self_attention", "conv_attention", "attention_gate").
            attn_args: Keyword arguments for an Attention module
              `from chuchichaestli.models.attention`
            kwargs: Additional keyword arguments for the convolutional layer.
        """
        super().__init__()
        self.norm: nn.Module | None = None
        self.act: nn.Module | None = None
        self.attn: nn.Module | None = None
        self.dropout: nn.Module | None = None
        self.norm_first = norm_first
        self.act_last = act_last
        if norm and norm_type is not None:
            if norm_type == "group" and (
                in_channels % num_groups != 0 or in_channels < num_groups
            ):
                if in_channels % 2 == 0:
                    num_groups = in_channels // 2
                else:
                    num_groups = gcd(in_channels, in_channels // 3)
            self.norm = Norm(
                dimensions,
                norm_type,
                in_channels if norm_first else out_channels,
                num_groups,
            )
        if act and act_fn is not None:
            self.act = ACTIVATION_FUNCTIONS[act_fn]()
        if dropout and dropout_p is not None and dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)
        match attention:
            case "self_attention" | "conv_attention":
                self.attn = ATTENTION_MAP[attention](
                    dimensions, in_channels, **attn_args
                )
            case "attention_gate":
                self.attn = ATTENTION_MAP[attention](
                    in_channels, out_channels, **attn_args
                )
            case _:
                self.attn = None
        self.conv = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

        if double_conv:
            self.conv2 = DIM_TO_CONV_MAP[dimensions](
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding="same",
            )

    def forward(self, x: torch.Tensor, _h: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        h = x
        if self.norm_first:
            h = self.norm(h) if self.norm is not None else h
        if not self.act_last:
            h = self.act(h) if self.act is not None else h
        h = self.dropout(h) if self.dropout is not None else h
        h = self.attn(h, _h if _h is not None else h) if self.attn else h
        h = self.conv(h)
        h = self.conv2(h) if hasattr(self, "conv2") else h
        if not self.norm_first:
            h = self.norm(h) if self.norm is not None else h
        if self.act_last:
            h = self.act(h) if self.act is not None else h
        return h


class ResidualBlock(nn.Module):
    """Residual block (with 2 convolutional layers) including a skip connection.

    Includes (in following order):
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - convolution (default: 3x3, stride 1)
        - time embedding (optional; default: `False`)
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - dropout (optional; default: `0.1`)
        - convolution (default: 3x3, stride 1)
        - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    @alias_kwargs(
        {
            "res_groups": "num_groups",
            "res_act_fn": "act_fn",
            "res_dropout": "dropout_p",
            "res_norm_type": "norm_type",
            "res_kernel_size": "kernel_size",
            "res_stride": "stride",
            "res_bias": "bias",
        }
    )
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = False,
        time_channels: int = 32,
        res_groups: int = 32,
        res_act_fn: str = "silu",
        res_dropout: float = 0.1,
        res_norm_type: str = "group",
        res_kernel_size: int = 3,
        res_stride: int = 1,
        res_bias: bool = True,
        **kwargs,
    ):
        """Initialize the residual block.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embedding: Whether to accept time embedding input.
            time_channels: Channels of the time embedding.
            res_groups: Number of groups for the residual block normalization,
                if `norm_type == 'group'`.
            res_act_fn: Activation function.
            res_dropout: Dropout probability.
            res_norm_type: Normalization type.
            res_kernel_size: Kernel size for the residual block.
            res_stride: Stride of the residual block.
            res_bias: Bias for convolutional layers.
            num_groups: Number of groups; alternative input for `res_groups`.
            act_fn: Activation function; alternative input for `res_act_fn`.
            norm_type: Normalization type; alternative input for `res_norm_type`.
            dropout_p: Dropout probability; alternative input for `res_dropout`.
            kernel_size: Kernel size for the residual block;
                alternative input for `res_kernel_size`.
            stride: Stride of the residual block; alternative input for `res_stride`.
            bias: Bias for convolutions; alternative input for `res_bias`.
            kwargs: Additional or alternative keyword arguments.
        """
        super().__init__()
        act_cls = ACTIVATION_FUNCTIONS[res_act_fn]
        conv_cls = DIM_TO_CONV_MAP[dimensions]

        self.dimensions = dimensions

        self.norm1 = Norm(dimensions, res_norm_type, in_channels, res_groups)
        self.act1 = act_cls()
        self.conv1 = conv_cls(
            in_channels,
            out_channels,
            kernel_size=res_kernel_size,
            padding="same" if res_stride == 1 else (res_kernel_size - 1) // 2,
            stride=res_stride,
            bias=res_bias,
        )

        self.norm2 = Norm(dimensions, res_norm_type, out_channels, res_groups)
        self.act2 = act_cls()
        self.conv2 = conv_cls(
            out_channels,
            out_channels,
            kernel_size=res_kernel_size,
            padding="same",
            bias=res_bias,
        )

        self.shortcut = (
            conv_cls(in_channels, out_channels, kernel_size=1, stride=res_stride, bias=res_bias)
            if in_channels != out_channels or res_stride != 1
            else nn.Identity()
        )

        self.time_embedding = time_embedding
        if time_embedding:
            self.time_proj = nn.Linear(time_channels, out_channels)
            self.time_act = act_cls()

        self.dropout = nn.Dropout(res_dropout) if res_dropout > 0 else None

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the residual block."""
        hh = self.conv1(self.act1(self.norm1(x)))
        idx = [slice(None), slice(None)] + [None] * self.dimensions
        if self.time_embedding:
            hh += self.time_proj(self.time_act(t))[tuple(idx)]
        hh = self.act2(self.norm2(hh))
        hh = self.dropout(hh) if self.dropout is not None else hh
        hh = self.conv2(hh)

        return hh + self.shortcut(x)


class ResidualBottleneck(nn.Module):
    """Residual bottleneck block (with 3 convolutional layers) including a skip connection.

    Includes (in following order):
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - convolution (default: 1x1, stride 1)
        - time embedding (optional; default: `False`)
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - convolution (default: 3x3, stride 1)
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - dropout (optional; default: `0.1`)
        - convolution (default: 1x1, stride 1)
        - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    @alias_kwargs(
        {
            "res_groups": "num_groups",
            "res_act_fn": "act_fn",
            "res_dropout": "dropout_p",
            "res_norm_type": "norm_type",
            "res_kernel_size": "kernel_size",
            "res_stride": "stride",
            "res_bias": "bias",
            "res_expansion": "expansion",
        }
    )
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = False,
        time_channels: int = 32,
        res_groups: int = 32,
        res_act_fn: ActivationTypes = "silu",
        res_dropout: float = 0.1,
        res_norm_type: NormTypes = "group",
        res_kernel_size: int = 3,
        res_stride: int = 1,
        res_bias: bool = True,
        res_expansion: int = 4,
        **kwargs,
    ):
        """Initialize the residual block.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embedding: Whether to accept time embedding input.
            time_channels: Channels of the time embedding.
            res_groups: Number of groups for the residual block normalization,
                if `norm_type == 'group'`.
            res_act_fn: Activation function.
            res_dropout: Dropout probability.
            res_norm_type: Normalization type.
            res_kernel_size: Kernel size for the residual block.
            res_stride: Stride of the residual block (applied in the mid-layer).
            res_bias: Bias for convolutional layers.
            res_expansion: Factor for the channel number increase from the mid-layer.
            num_groups: Number of groups; alternative input for `res_groups`.
            act_fn: Activation function; alternative input for `res_act_fn`.
            norm_type: Normalization type; alternative input for `res_norm_type`.
            dropout_p: Dropout probability; alternative input for `res_dropout`.
            kernel_size: Kernel size for the residual block;
                alternative input for `res_kernel_size`.
            stride: Stride of the residual block; alternative input for `res_stride`.
            bias: Bias for convolutional layers; alternative input for `res_bias`.
            expansion: Factor for the channel number increase from the mid-layer;
                alternative input for `res_expansion`.
            kwargs: Additional or alternative keyword arguments.
        """
        super().__init__()
        act_cls = ACTIVATION_FUNCTIONS[res_act_fn]
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        n_channels = out_channels // res_expansion

        self.dimensions = dimensions

        self.norm1 = Norm(dimensions, res_norm_type, in_channels, res_groups)
        self.act1 = act_cls()
        self.conv1 = conv_cls(in_channels, n_channels, kernel_size=1, bias=res_bias)

        self.norm2 = Norm(
            dimensions, res_norm_type, n_channels, res_groups // res_expansion
        )
        self.act2 = act_cls()
        self.conv2 = conv_cls(
            n_channels,
            n_channels,
            kernel_size=res_kernel_size,
            padding="same" if res_stride == 1 else (res_kernel_size - 1) // 2,
            stride=res_stride,
            bias=res_bias,
        )

        self.norm3 = Norm(
            dimensions, res_norm_type, n_channels, res_groups // res_expansion
        )
        self.act3 = act_cls()
        self.conv3 = conv_cls(n_channels, out_channels, kernel_size=1, bias=res_bias)

        self.shortcut = (
            conv_cls(in_channels, out_channels, kernel_size=1, stride=res_stride, bias=res_bias)
            if in_channels != out_channels or res_stride != 1
            else nn.Identity()
        )

        self.time_embedding = time_embedding
        if time_embedding:
            self.time_proj = nn.Linear(time_channels, out_channels)
            self.time_act = act_cls()

        self.dropout = nn.Dropout(res_dropout) if res_dropout > 0 else None

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the residual block."""
        hh = self.conv1(self.act1(self.norm1(x)))
        idx = [slice(None), slice(None)] + [None] * self.dimensions
        if self.time_embedding:
            hh += self.time_proj(self.time_act(t))[tuple(idx)]
        hh = self.conv2(self.act2(self.norm2(hh)))
        hh = self.act3(self.norm3(hh))
        hh = self.dropout(hh) if self.dropout is not None else hh
        hh = self.conv3(hh)
        return hh + self.shortcut(x)


class DownBlock(nn.Module):
    """Standard block for the encoder of a U-Net (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `None`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = False,
        time_channels: int = 32,
        res_args: dict = {},
        attention: AttentionDownTypes | None = None,
        attn_args: dict = {},
    ):
        """Initialize the down block."""
        super().__init__()
        self.res_block = ResidualBlock(
            dimensions,
            in_channels,
            out_channels,
            time_embedding,
            time_channels,
            **res_args,
        )

        match attention:
            case "self_attention" | "conv_attention":
                self.attn = ATTENTION_MAP[attention](
                    dimensions, in_channels, **attn_args
                )
            case _:
                self.attn = None

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the down block."""
        x = self.attn(x, None) if self.attn else x
        x = self.res_block(x, t)
        return x


class AutoencoderDownBlock(DownBlock):
    """Standard block for the encoder of an autoencoder (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `None`)
        - Residual block (x1 or x2):
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        res_args: dict = {},
        attention: AttentionDownTypes | None = None,
        attn_args: dict = {},
    ):
        """Initialize the encoder down block."""
        super().__init__(
            dimensions,
            in_channels,
            out_channels,
            False,
            None,
            res_args,
            attention,
            attn_args,
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the encoder down block."""
        return super().forward(x)


class MidBlock(nn.Module):
    """Standard block for the bottleneck of a U-Net (meant to keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `None`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    def __init__(
        self,
        dimensions: int,
        channels: int,
        time_embedding: bool = False,
        time_channels: int = 32,
        res_args: dict = {},
        attention: AttentionDownTypes | None = None,
        attn_args: dict = {},
    ):
        """Initialize the mid block."""
        super().__init__()
        self.res_block = ResidualBlock(
            dimensions,
            channels,
            channels,
            time_embedding,
            time_channels,
            **res_args,
        )
        match attention:
            case "self_attention" | "conv_attention":
                self.attn = ATTENTION_MAP[attention](dimensions, channels, **attn_args)
            case _:
                self.attn = None

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the mid block."""
        x = self.attn(x, None) if self.attn else x
        x = self.res_block(x, t)
        return x


class AutoencoderMidBlock(MidBlock):
    """Standard block for the bottleneck of an autoencoder (meant to keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `None`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    def __init__(
        self,
        dimensions: int,
        channels: int,
        res_args: dict = {},
        attention: AttentionDownTypes | None = None,
        attn_args: dict = {},
    ):
        """Initialize the encoder mid block."""
        super().__init__(
            dimensions,
            channels,
            False,
            None,
            res_args,
            attention,
            attn_args,
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the encoder mid block."""
        return super().forward(x)


class UpBlock(nn.Module):
    """Standard block for the decoder of a U-Net (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `None`)
        - Residual block:
            - skip connection from encoder (default: via concatenation with the input)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        time_embedding: bool = False,
        time_channels: int = 32,
        res_args: dict = {},
        attention: AttentionTypes | None = None,
        attn_args: dict = {},
        skip_connection_action: Literal["concat", "avg", "add"] | None = None,
    ):
        """Initialize the up block."""
        super().__init__()
        self.skip_connection_action = skip_connection_action
        if skip_connection_action == "concat":
            self.res_block = ResidualBlock(
                dimensions,
                in_channels + in_channels,
                out_channels,
                time_embedding,
                time_channels,
                **res_args,
            )
        elif skip_connection_action in ["avg", "add", None]:
            self.res_block = ResidualBlock(
                dimensions,
                in_channels,
                out_channels,
                time_embedding,
                time_channels,
                **res_args,
            )
        else:
            raise ValueError(
                f"Invalid skip connection action: {skip_connection_action}"
            )

        match attention:
            case "self_attention" | "conv_attention":
                self.attn = ATTENTION_MAP[attention](
                    dimensions, in_channels, **attn_args
                )
            case "attention_gate":
                self.attn = ATTENTION_MAP[attention](
                    dimensions, in_channels, in_channels, **attn_args
                )
            case _:
                self.attn = None

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the up block."""
        x = self.attn(x, h) if self.attn else x
        if self.skip_connection_action == "avg":
            replication_factor = x.shape[1] // h.shape[1]
            h = h.repeat(
                (1, replication_factor) + (1,) * len(x.shape[2:])
            )  # Repeat channels
            xh = (x + h) / 2
        elif self.skip_connection_action == "add":
            replication_factor = x.shape[1] // h.shape[1]
            h = h.repeat(
                (1, replication_factor) + (1,) * len(x.shape[2:])
            )  # Repeat channels
            xh = x + h
        elif self.skip_connection_action == "concat":
            xh = torch.cat([x, h], dim=1)
        else:
            xh = x
        x = self.res_block(xh, t)
        return x


class AutoencoderUpBlock(nn.Module):
    """Standard block for the decoder of an autoencoder (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `None`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        res_args: dict = {},
        attention: Literal["self_attention", "conv_attention"] | None = None,
        attn_args: dict = {},
    ):
        """Initialize the up block."""
        super().__init__()
        self.res_block = ResidualBlock(
            dimensions,
            in_channels,
            out_channels,
            False,
            None,
            **res_args,
        )

        match attention:
            case "self_attention" | "conv_attention":
                self.attn = ATTENTION_MAP[attention](
                    dimensions, in_channels, **attn_args
                )
            case "attention_gate":
                self.attn = ATTENTION_MAP[attention](
                    in_channels, out_channels, **attn_args
                )
            case _:
                self.attn = None

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the up block."""
        x = self.attn(x, x) if self.attn else x
        x = self.res_block(x)
        return x


# Encoding / Decoding blocks for U-Net/Autoencoder
AttnDownBlock = partialclass(
    "AttnDownBlock",
    DownBlock,
    attention="self_attention",
    __doc__="""
    Attention block for the encoder of a U-Net (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'self_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
AttnMidBlock = partialclass(
    "AttnMidBlock",
    MidBlock,
    attention="self_attention",
    __doc__="""
    Attention block for the bottleneck of a U-Net (meant to keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'self_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
AttnUpBlock = partialclass(
    "AttnUpBlock",
    UpBlock,
    attention="self_attention",
    __doc__="""
    Attention block for the decoder of a U-Net (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'self_attention'`)
        - Residual block:
            - skip connection from encoder (default: via concatenation with the input)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
ConvAttnDownBlock = partialclass(
    "ConvAttnDownBlock",
    DownBlock,
    attention="conv_attention",
    __doc__="""
    Convolutional attention block for the encoder of a U-Net (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'conv_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
ConvAttnMidBlock = partialclass(
    "ConvAttnMidBlock",
    MidBlock,
    attention="conv_attention",
    __doc__="""
    Convolutional attention block for the bottleneck of a U-Net (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'conv_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
ConvAttnUpBlock = partialclass(
    "ConvAttnUpBlock",
    UpBlock,
    attention="conv_attention",
    __doc__="""
    Convolutional attention block for the decoder of a U-Net (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (optional; default: `'conv_attention'`)
        - Residual block:
            - skip connection from encoder (default: via concatenation with the input)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
AttnGateUpBlock = partialclass(
    "AttnGateUpBlock",
    UpBlock,
    attention="attention_gate",
    __doc__="""
    Gated attention block for the decoder of a U-Net (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'attention_gate'`, merges skip connection from the encoder)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - time embedding (optional; default: `False`)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
AttnAutoencoderDownBlock = partialclass(
    "AttnAutoencoderDownBlock",
    AutoencoderDownBlock,
    attention="self_attention",
    __doc__="""
    Attention block for the encoder of an autoencoder (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'self_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
AttnAutoencoderMidBlock = partialclass(
    "AttnAutoencoderMidBlock",
    AutoencoderMidBlock,
    attention="self_attention",
    __doc__="""
    Attention block for the bottleneck of an autoencoder (meant to keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'self_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
AttnAutoencoderUpBlock = partialclass(
    "AttnAutoencoderUpBlock",
    AutoencoderUpBlock,
    attention="self_attention",
    __doc__="""
    Attention block for the decoder of an autoencoder (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'self_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
ConvAttnAutoencoderDownBlock = partialclass(
    "ConvAttnAutoencoderDownBlock",
    AutoencoderDownBlock,
    attention="conv_attention",
    __doc__="""
    Convolutional attention block for the encoder of an autoencoder (meant to increase/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'conv_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
ConvAttnAutoencoderMidBlock = partialclass(
    "ConvAttnAutoencoderMidBlock",
    AutoencoderMidBlock,
    attention="conv_attention",
    __doc__="""
    Convolutional attention block for the bottleneck of an autoencoder (meant to keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'conv_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
ConvAttnAutoencoderUpBlock = partialclass(
    "ConvAttnAutoencoderUpBlock",
    AutoencoderUpBlock,
    attention="conv_attention",
    __doc__="""
    Convolutional attention block for the decoder of an autoencoder (meant to decrease/keep channel dimension).

    Includes (in following order):
        - Attention layer (default: `'conv_attention'`)
        - Residual block:
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - convolution (default: 3x3, stride 1)
            - normalization (default: `'group'`)
            - activation (default: `'silu'`)
            - dropout (optional; default: `0.1`)
            - convolution (default: 3x3, stride 1)
            - residual shortcut (with 1x1 conv if input and output channels differ)
    """,
)
EncoderOutBlock = partialclass(
    "EncoderOutBlock",
    BaseConvBlock,
    act_fn="silu",
    norm_type="group",
    num_groups=8,
    kernel_size=3,
    stride=1,
    padding="same",
    __doc__="""
    Convolutional output block for the encoder of an autoencoder.

    Includes:
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - convolution (default: 3x3, stride 1)
    """,
)
VAEEncoderOutBlock = partialclass(
    "VAEEncoderOutBlock",
    BaseConvBlock,
    act_fn="silu",
    norm_type="group",
    num_groups=8,
    kernel_size=3,
    stride=1,
    padding="same",
    double_conv=True,
    __doc__="""
    Convolutional output block for a VAE encoder.

    Includes:
        - normalization (default: `'group'`)
        - activation (default: `'silu'`)
        - convolution (default: 3x3, stride 1)
        - convolution (default: 1x1, stride 1)
    """,
)
DecoderInBlock = partialclass(
    "DecoderInBlock",
    BaseConvBlock,
    kernel_size=3,
    stride=1,
    padding="same",
    __doc__="""
    Convolutional input block for the decoder of an autoencoder.

    Includes:
        - convolution (default: 3x3, stride 1)
    """,
)
VAEDecoderInBlock = partialclass(
    "DecoderInBlock",
    BaseConvBlock,
    kernel_size=3,
    stride=1,
    padding="same",
    double_conv=True,
    __doc__="""
    Convolutional input block for the decoder of an autoencoder.

    Includes:
        - convolution (default: 3x3, stride 1)
        - convolution (default: 1x1, stride 1)
    """,
)


# Convolutional blocks for simple CNNs
ConvDownBlock = partialclass(
    "ConvDownBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    __doc__="""
    Convolutional block without normalization or non-linearity.

    Includes:
        - dropout (optional)
        - convolution (4x4, stride 2)
    """,
)
ConvDownsampleBlock = partialclass(
    "ConvDownsampleBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    padding=2,
    __doc__="""
    Convolutional block without normalization or non-linearity.

    Includes:
        - dropout (optional)
        - convolution (4x4, stride 2, padding 2)
    """,
)
AttnConvDownBlock = partialclass(
    "AttnConvDownBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    attention="self_attention",
    __doc__="""
    Convolutional block without normalization or non-linearity.

    Includes:
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2)
    """,
)
AttnConvDownsampleBlock = partialclass(
    "AttnConvDownsampleBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    attention="self_attention",
    padding=2,
    __doc__="""
    Convolutional block without normalization or non-linearity.

    Includes:
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2, padding 2)
    """,
)
ConvBlock = partialclass(
    "ConvBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    stride=1,
    __doc__="""
    Convolutional block without normalization or non-linearity.

    Includes:
        - dropout (optional)
        - convolution (4x4, stride 1)
    """,
)
AttnConvBlock = partialclass(
    "AttnConvBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    stride=1,
    attention="self_attention",
    __doc__="""
    Convolutional block without normalization or non-linearity.

    Includes:
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 1)
    """,
)
NormConvBlock = partialclass(
    "NormConvBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    stride=1,
    __doc__="""
    Convolutional block without non-linearity.

    Includes:
        - normalization (default: `'batch'`)
        - dropout (optional)
        - convolution (4x4, stride 1)
    """,
)
NormAttnConvBlock = partialclass(
    "NormAttnConvBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    stride=1,
    attention="self_attention",
    __doc__="""
    Convolutional block without non-linearity.

    Includes:
        - normalization (default: `'batch'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 1)
    """,
)
NormConvDownBlock = partialclass(
    "NormConvDownBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    __doc__="""
    Convolutional block without non-linearity.

    Includes:
        - normalization (default: `'batch'`)
        - dropout (optional)
        - convolution (4x4, stride 2)
    """,
)
NormConvDownsampleBlock = partialclass(
    "NormConvDownsampleBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    padding=2,
    __doc__="""
    Convolutional block without non-linearity.

    Includes:
        - normalization (default: `'batch'`)
        - dropout (optional)
        - convolution (4x4, stride 2, padding 2)
    """,
)
NormAttnConvDownBlock = partialclass(
    "NormAttnConvDownBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    attention="self_attention",
    __doc__="""
    Convolutional block without non-linearity.

    Includes:
        - normalization (default: `'batch'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2)
    """,
)
NormAttnConvDownsampleBlock = partialclass(
    "NormAttnConvDownsampleBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    attention="self_attention",
    padding=2,
    __doc__="""
    Convolutional block without non-linearity.

    Includes:
        - normalization (default: `'batch'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2, padding 2)
    """,
)
ActConvBlock = partialclass(
    "ActConvBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    stride=1,
    __doc__="""
    Convolutional block without normalization.

    Includes:
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - convolution (4x4, stride 1)
    """,
)
ActAttnConvBlock = partialclass(
    "ActAttnConvBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    stride=1,
    attention="self_attention",
    __doc__="""
    Convolutional block without normalization.

    Includes:
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 1)
    """,
)
ActConvDownBlock = partialclass(
    "ActConvDownBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    __doc__="""
    Convolutional block without normalization.

    Includes:
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - convolution (4x4, stride 2)
    """,
)
ActConvDownsampleBlock = partialclass(
    "ActConvDownsampleBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    padding=2,
    __doc__="""
    Convolutional block without normalization.

    Includes:
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - convolution (4x4, stride 2, padding 2)
    """,
)
ActAttnConvDownBlock = partialclass(
    "ActAttnConvDownBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    attention="self_attention",
    __doc__="""
    Convolutional block without normalization.

    Includes:
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2)
    """,
)
ActAttnConvDownsampleBlock = partialclass(
    "ActAttnConvDownsampleBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    attention="self_attention",
    padding=2,
    __doc__="""
    Convolutional block without normalization.

    Includes:
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2, padding 2)
    """,
)
NormActConvBlock = partialclass(
    "NormActConvBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    stride=1,
    __doc__="""
    Convolutional block.

    Includes:
        - normalization (default: `'batch'`)
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - convolution (4x4, stride 1)
    """,
)
NormActAttnConvBlock = partialclass(
    "NormActAttnConvBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    stride=1,
    attention="self_attention",
    __doc__="""
    Convolutional block.

    Includes:
        - normalization (default: `'batch'`)
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 1)
    """,
)
NormActConvDownBlock = partialclass(
    "NormActConvDownBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    __doc__="""
    Convolutional block.

    Includes:
        - normalization (default: `'batch'`)
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - convolution (4x4, stride 2)
    """,
)
NormActConvDownsampleBlock = partialclass(
    "NormActConvDownsampleBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    padding=2,
    __doc__="""
    Convolutional block.

    Includes:
        - normalization (default: `'batch'`)
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - convolution (4x4, stride 2, padding)
    """,
)
NormActAttnConvDownBlock = partialclass(
    "NormActAttnConvDownBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    attention="self_attention",
    __doc__="""
    Convolutional block.

    Includes:
        - normalization (default: `'batch'`)
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2)
    """,
)
NormActAttnConvDownsampleBlock = partialclass(
    "NormActAttnConvDownsampleBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    attention="self_attention",
    padding=2,
    __doc__="""
    Convolutional block.

    Includes:
        - normalization (default: `'batch'`)
        - activation (default: `'leakyrelu,0.2'`)
        - dropout (optional)
        - attention (default: `'self_attention'`)
        - convolution (4x4, stride 2, padding 2)
    """,
)


BLOCK_MAP: dict[str, Callable] = {
    # U-Net blocks
    "DownBlock": DownBlock,
    "MidBlock": MidBlock,
    "UpBlock": UpBlock,
    "AttnDownBlock": AttnDownBlock,
    "AttnMidBlock": AttnMidBlock,
    "AttnUpBlock": AttnUpBlock,
    "AttnGateUpBlock": AttnGateUpBlock,
    "ConvAttnDownBlock": ConvAttnDownBlock,
    "ConvAttnMidBlock": ConvAttnMidBlock,
    "ConvAttnUpBlock": ConvAttnUpBlock,
    # Autoencoder blocks
    "AutoencoderDownBlock": AutoencoderDownBlock,
    "AutoencoderMidBlock": AutoencoderMidBlock,
    "AutoencoderUpBlock": AutoencoderUpBlock,
    "AttnAutoencoderDownBlock": AttnAutoencoderDownBlock,
    "AttnAutoencoderMidBlock": AttnAutoencoderMidBlock,
    "AttnAutoencoderUpBlock": AttnAutoencoderUpBlock,
    "ConvAttnAutoencoderDownBlock": ConvAttnAutoencoderDownBlock,
    "ConvAttnAutoencoderMidBlock": ConvAttnAutoencoderMidBlock,
    "ConvAttnAutoencoderUpBlock": ConvAttnAutoencoderUpBlock,
    "EncoderOutBlock": EncoderOutBlock,
    "VAEEncoderOutBlock": VAEEncoderOutBlock,
    "DecoderInBlock": DecoderInBlock,
    "VAEDecoderInBlock": VAEDecoderInBlock,
}


CONV_BLOCK_MAP: dict[str, Callable] = {
    "ResidualBlock": ResidualBlock,
    "ResidualBottleneck": ResidualBottleneck,
    "ConvDownBlock": ConvDownBlock,
    "ConvDownsampleBlock": ConvDownsampleBlock,
    "AttnConvDownBlock": AttnConvDownBlock,
    "AttnConvDownsampleBlock": AttnConvDownsampleBlock,
    "ConvBlock": ConvBlock,
    "AttnConvBlock": AttnConvBlock,
    "NormConvBlock": NormConvBlock,
    "NormAttnConvBlock": NormAttnConvBlock,
    "NormConvDownBlock": NormConvDownBlock,
    "NormConvDownsampleBlock": NormConvDownsampleBlock,
    "NormAttnConvDownBlock": NormAttnConvDownBlock,
    "NormAttnConvDownsampleBlock": NormAttnConvDownsampleBlock,
    "ActConvBlock": ActConvBlock,
    "ActAttnConvBlock": ActAttnConvBlock,
    "ActConvDownBlock": ActConvDownBlock,
    "ActConvDownsampleBlock": ActConvDownsampleBlock,
    "ActAttnConvDownBlock": ActAttnConvDownBlock,
    "ActAttnConvDownsampleBlock": ActAttnConvDownsampleBlock,
    "NormActConvBlock": NormActConvBlock,
    "NormActAttnConvBlock": NormActAttnConvBlock,
    "NormActConvDownBlock": NormActConvDownBlock,
    "NormActConvDownsampleBlock": NormActConvDownsampleBlock,
    "NormActAttnConvDownBlock": NormActAttnConvDownBlock,
    "NormActAttnConvDownsampleBlock": NormActAttnConvDownsampleBlock,
}


RESIDUAL_BLOCK_MAP = {
    "ResidualBlock": ResidualBlock,
    "ResidualBottleneck": ResidualBottleneck,
}

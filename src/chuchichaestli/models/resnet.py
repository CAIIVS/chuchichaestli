# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""ResNet block implementation."""

import torch
from torch import nn
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.blocks import (
    NormActConvDownBlock,
    ResidualBlock,
    ResidualBottleneck,
    RESIDUAL_BLOCK_MAP,
)
from chuchichaestli.models.downsampling import DOWNSAMPLE_FUNCTIONS
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.utils import partialclass
from typing import Literal
from collections.abc import Sequence


class ResNet(nn.Module):
    """Generic ResNet model."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        num_layers: Sequence[int],
        channel_mults: Sequence[int] | None = None,
        n_channels: int = 64,
        block_type: Literal[
            "ResidualBlock", "ResidualBottleneck"
        ] = "ResidualBottleneck",
        act_fn: ActivationTypes = "relu",
        norm_type: NormTypes = "batch",
        dropout_p: float = 0,
        bias: bool = False,
        pool_in_type: Literal["MaxPool", "AdaptiveMaxPool"] = "MaxPool",
        pool_out_type: Literal["AvgPool", "AdaptiveAvgPool"] = "AdaptiveAvgPool",
        zero_init: bool = True,
    ):
        """Initialize a ResNet.

        Args:
            dimensions: Number of (spatial) dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_layers: Number of block repetitions in each layer.
            channel_mults: Factor to increase number of channels in each layer.
            n_channels: Number of channels in the first block.
            block_type: Type of residual block in the network layers.
            act_fn: Activation function.
            norm_type: Normalization type for the residual blocks.
            dropout_p: Dropout probability of the residual blocks.
            bias: Bias for convolutional layers.
            pool_in_type: Type of pooling layer for the input layer
              (default `"MaxPool"`).
            pool_out_type: Type of pooling layer for the output layer
              (default `"AdaptiveAvgPool"`).
            zero_init: If `True`, the residual block's last normalization layers
              are zero-initialized following https://arxiv.org/abs/1706.02677 (Section 5.1).
        """
        super().__init__()
        if channel_mults is None:
            channel_mults = (1,) if block_type == "ResidualBlock" else (4,)
            channel_mults += (2,) * (len(num_layers) - 1)
        if len(channel_mults) < len(num_layers):
            raise ValueError(
                f"channel_mults needs to be same length as `num_layers`,"
                f" got {len(channel_mults)} instead of {len(num_layers)}."
            )
        block_cls = RESIDUAL_BLOCK_MAP[block_type]
        pool_in_cls = DOWNSAMPLE_FUNCTIONS[pool_in_type]
        pool_out_cls = DOWNSAMPLE_FUNCTIONS[pool_out_type]
        self.conv_in = NormActConvDownBlock(
            dimensions,
            in_channels,
            n_channels,
            kernel_size=7,
            padding=3,
            norm_type=norm_type,
            norm_first=False,
            dropout=dropout_p,
            act_fn=act_fn,
            act_last=True,
            bias=bias,
        )
        self.pool_in = pool_in_cls(dimensions)
        self.layers = nn.ModuleList([])
        for i, (n_layer, c_mult) in enumerate(zip(num_layers, channel_mults)):
            blocks = [
                block_cls(
                    dimensions,
                    n_channels,
                    (n_channels := c_mult * n_channels),
                    stride=1 + int(i > 0),
                    act_fn=act_fn,
                    norm_type=norm_type,
                    dropout_p=dropout_p,
                    bias=bias,
                )
            ]
            for _ in range(1, n_layer):
                blocks.append(
                    block_cls(
                        dimensions,
                        n_channels,
                        n_channels,
                        act_fn=act_fn,
                        norm_type=norm_type,
                        dropout_p=dropout_p,
                        bias=bias,
                    )
                )
            self.layers.append(nn.Sequential(*blocks))
        self.pool_out = pool_out_cls(dimensions)
        self.fc_out = nn.Linear(n_channels, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=act_fn)
            elif isinstance(
                m,
                nn.GroupNorm
                | nn.BatchNorm1d
                | nn.BatchNorm2d
                | nn.BatchNorm3d
                | nn.InstanceNorm1d
                | nn.InstanceNorm2d
                | nn.InstanceNorm3d,
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if zero_init:
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.norm2.norm.weight, 0)
                if isinstance(m, ResidualBottleneck):
                    nn.init.constant_(m.norm3.norm.weight, 0)

    def forward(self, x: torch.Tensor, *args):
        """Forward pass through the ResNet."""
        h = self.conv_in(x)
        h = self.pool_in(h)
        for layer in self.layers:
            h = layer(h)
        h = self.pool_out(h)
        h = torch.flatten(h, 1)
        h = self.fc_out(h)
        return h


ResNet18 = partialclass(
    "ResNet18",
    ResNet,
    num_layers=[2, 2, 2, 2],
    channel_mults=[1, 2, 2, 2],
    block_type="ResidualBlock",
    __doc__="""ResNet with 18 convolutional layers.""",
)

ResNet34 = partialclass(
    "ResNet34",
    ResNet,
    num_layers=[3, 4, 6, 3],
    block_type="ResidualBlock",
    __doc__="""ResNet with 34 convolutional layers.""",
)

ResNet50 = partialclass(
    "ResNet50",
    ResNet,
    num_layers=[3, 4, 6, 3],
    block_type="ResidualBottleneck",
    __doc__="""ResNet with 50 convolutional layers.""",
)

ResNet101 = partialclass(
    "ResNet101",
    ResNet,
    num_layers=[3, 4, 23, 3],
    block_type="ResidualBottleneck",
    __doc__="""ResNet with 101 convolutional layers.""",
)

ResNet152 = partialclass(
    "ResNet152",
    ResNet,
    num_layers=[3, 8, 36, 3],
    block_type="ResidualBottleneck",
    __doc__="""ResNet with 152 convolutional layers.""",
)

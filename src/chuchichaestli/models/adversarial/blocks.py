"""Blocks for adversarial models.

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

from math import gcd
import torch
from torch import nn
from chuchichaestli.models.norm import Norm
from chuchichaestli.models.resnet import ResidualBlock as ResnetBlock
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.attention import ATTENTION_MAP
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.utils import partialclass
from collections.abc import Callable


__all__ = [
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
    "ResidualBlock",  # subclassed from chuchichaestli.models.resent.ResidualBlock
]


class BaseConvBlock(nn.Module):
    """Convolutional block with various components.

    Components include (in following order):
          - normalization (optional)
          - activation (optional)
          - dropout (optional)
          - attention (optional)
          - convolution
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        act: bool = True,
        act_fn: str | None = None,
        norm: bool = True,
        norm_type: str | None = None,
        num_groups: int = 16,
        dropout: bool = True,
        dropout_p: float | None = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        attention: str = "",
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
          norm: Use normalization in the block.
          norm_type: Normalization type for the conv blocks.
          dropout: Use dropout in the block.
          dropout_p: Dropout probability of the conv blocks.
          num_groups: Number of groups for the conv block normlization (if norm_type == 'group').
          kernel_size: Kernel size for the conv block.
          stride: Stride for the conv block.
          padding: Padding size for the conv block.
          attention: Attention descriptor from {"self_attention", "attention_gate"};
            if None or unknown, no attention is used.
          attn_args: Keyword arguments for a `SelfAttention` module
            `from chuchichaestli.models.attention.self_attention`
          kwargs: Additional keyword arguments for the convolutional layer.
        """
        super().__init__()
        self.norm: nn.Module | None = None
        self.act: nn.Module | None = None
        self.attn: nn.Module | None = None
        self.dropout: nn.Module | None = None
        if norm and norm_type is not None:
            if norm_type == "group" and (
                in_channels % num_groups != 0 or in_channels < num_groups
            ):
                if in_channels % 2 == 0:
                    num_groups = in_channels // 2
                else:
                    num_groups = gcd(in_channels, in_channels // 3)
            self.norm = Norm(dimensions, norm_type, in_channels, num_groups)
        if act and act_fn is not None:
            self.act = ACTIVATION_FUNCTIONS[act_fn]()
        if dropout and dropout_p is not None and dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)
        match attention:
            case "self_attention":
                self.attn = ATTENTION_MAP[attention](in_channels, **attn_args)
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

    def forward(self, x: torch.Tensor, _h: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the convolutional downsampling block."""
        h = x
        h = self.norm(h) if self.norm is not None else h
        h = self.act(h) if self.act is not None else h
        h = self.dropout(h) if self.dropout is not None else h
        h = self.attn(h, _h if _h is not None else h) if self.attn else h
        return self.conv(h)


ConvDownBlock = partialclass("ConvDownBlock", BaseConvBlock, act=False, norm=False)

ConvDownsampleBlock = partialclass(
    "ConvDownsampleBlock", BaseConvBlock, act=False, norm=False, padding=2
)

AttnConvDownBlock = partialclass(
    "AttnConvDownBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    attention="self_attention",
)

AttnConvDownsampleBlock = partialclass(
    "AttnConvDownsampleBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    attention="self_attention",
    padding=2,
)

ConvBlock = partialclass("ConvBlock", BaseConvBlock, act=False, norm=False, stride=1)

AttnConvBlock = partialclass(
    "AttnConvBlock",
    BaseConvBlock,
    act=False,
    norm=False,
    stride=1,
    attention="self_attention",
)

NormConvBlock = partialclass(
    "NormConvBlock", BaseConvBlock, act=False, norm_type="batch", stride=1
)

NormAttnConvBlock = partialclass(
    "NormAttnConvBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    stride=1,
    attention="self_attention",
)

NormConvDownBlock = partialclass(
    "NormConvDownBlock", BaseConvBlock, act=False, norm_type="batch"
)

NormConvDownsampleBlock = partialclass(
    "NormConvDownsampleBlock", BaseConvBlock, act=False, norm_type="batch", padding=2
)

NormAttnConvDownBlock = partialclass(
    "NormAttnConvDownBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    attention="self_attention",
)

NormAttnConvDownsampleBlock = partialclass(
    "NormAttnConvDownsampleBlock",
    BaseConvBlock,
    act=False,
    norm_type="batch",
    attention="self_attention",
)

ActConvBlock = partialclass(
    "ActConvBlock", BaseConvBlock, norm=False, act_fn="leakyrelu,0.2", stride=1
)

ActAttnConvBlock = partialclass(
    "ActAttnConvBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    stride=1,
    attention="self_attention",
)

ActConvDownBlock = partialclass(
    "ActConvDownBlock", BaseConvBlock, norm=False, act_fn="leakyrelu,0.2"
)

ActConvDownsampleBlock = partialclass(
    "ActConvDownsampleBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    padding=2,
)

ActAttnConvDownBlock = partialclass(
    "ActAttnConvDownBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    attention="self_attention",
)

ActAttnConvDownsampleBlock = partialclass(
    "ActAttnConvDownsampleBlock",
    BaseConvBlock,
    norm=False,
    act_fn="leakyrelu,0.2",
    attention="self_attention",
    padding=2,
)

NormActConvBlock = partialclass(
    "NormActConvBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    stride=1,
)

NormActAttnConvBlock = partialclass(
    "NormActAttnConvBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    stride=1,
    attention="self_attention",
)

NormActConvDownBlock = partialclass(
    "NormActConvDownBlock", BaseConvBlock, norm_type="batch", act_fn="leakyrelu,0.2"
)

NormActConvDownsampleBlock = partialclass(
    "NormActConvDownsampleBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    padding=2,
)

NormActAttnConvDownBlock = partialclass(
    "NormActAttnConvDownBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    attention="self_attention",
)

NormActAttnConvDownsampleBlock = partialclass(
    "NormActAttnConvDownsampleBlock",
    BaseConvBlock,
    norm_type="batch",
    act_fn="leakyrelu,0.2",
    attention="self_attention",
    padding=2,
)


class ResidualBlock(ResnetBlock):
    """Residual convolutional block with skip connections.

    Same implementation as chuchichaestli.models.resnet.ResidualBlock but with
    different argument keys and default values (analogous to BaseConvBlock).
    The time embedding is removed by default.
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        act_fn: str = "silu",
        norm_type: str = "group",
        num_groups: int = 32,
        dropout_p: float = 0.0,
        kernel_size: int = 3,
        **kwargs,
    ):
        """Initialize the residual block.

        Args:
          dimensions: Number of dimensions.
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          act_fn: Activation function.
          norm_type: Normalization type for the conv blocks.
          dropout_p: Dropout probability of the conv blocks.
          num_groups: Number of groups for the conv block normlization (if norm_type == 'group').
          kernel_size: Kernel size for the conv block.
          kwargs: Additional keyword arguments.
        """
        kwargs["res_act_fn"] = act_fn
        kwargs["res_dropout"] = dropout_p
        kwargs["res_norm_type"] = norm_type
        kwargs["res_groups"] = num_groups
        kwargs["res_kernel_size"] = kernel_size
        for k in (
            "act",
            "dropout",
            "norm",
            "stride",
            "padding",
            "attention",
            "attn_args",
            "bias",
        ):
            if k in kwargs:
                kwargs.pop(k)
        super().__init__(dimensions, in_channels, out_channels, False, 0, **kwargs)

    def forward(self, x: torch.Tensor, _t: torch.Tensor = torch.empty(0)):
        """Forward pass through the residual block (time embedding optional)."""
        return super().forward(x, _t)


BLOCK_MAP: dict[str, Callable] = {
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
    "ResidualBlock": ResidualBlock,
}

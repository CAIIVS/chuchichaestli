"""PatchGAN discriminators.

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
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _AvgPoolNd, _MaxPoolNd
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.adversarial.blocks import BLOCK_MAP


__all__ = [
    "BlockDiscriminator",
    "PixelDiscriminator",
    "PatchDiscriminator",
    "AttnPatchDiscriminator",
    "AntialiasingDiscriminator",
    "AntialiasingPatchDiscriminator",
]


class BlockDiscriminator(nn.Sequential):
    """A base class for pixel and patch-based discriminators as in Pix2Pix.

    From the paper: https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        block_types: tuple[str, ...] = (
            "ConvDownBlock",
            "ActConvDownBlock",
            "NormActConvDownBlock",
            "NormActConvBlock",
            "NormActConvBlock",
        ),
        channel_mults: tuple[int, ...] | None = None,
        out_channels: int = 1,
        attn_n_heads: int = 1,
        attn_head_dim: int = 16,
        attn_gate_inter_channels: int = 32,
        **kwargs,
    ):
        """Construct a discriminator.

        Args:
          dimensions: Number of dimensions.
          in_channels: Number of input channels.
          n_channels: Number of channels in the first layer.
          block_types: Types of down blocks.
          channel_mults: Channel multipliers for each block.
          out_channels: Output channels (should generally be 1, i.e. true/fake).
          attn_n_heads: Number of attention heads.
          attn_head_dim: Dimension of the attention head.
          attn_gate_inter_channels: Number of intermediate channels for the attention gate.
          kwargs: Additional arguments for the blocks.
        """
        if dimensions not in DIM_TO_CONV_MAP:
            raise ValueError(
                f"Invalid number of dimensions ({dimensions}). "
                f"Must be one of {list(DIM_TO_CONV_MAP.keys())}."
            )

        if any(block_type not in BLOCK_MAP for block_type in block_types):
            raise ValueError(
                f"Invalid block types. Must be one of {list(BLOCK_MAP.keys())}."
            )

        n_blocks = len(block_types)
        if n_blocks < 2:
            raise ValueError(
                f"At least two convolutional blocks are required ({block_types} was given)."
            )

        if channel_mults is None:
            channel_mults = (2,) * (len(block_types) - 2)
        elif len(channel_mults) < n_blocks - 2:
            raise ValueError(
                f"Not enough channel multipliers. Must be at least len(block_types)-2 = {n_blocks-2}."
            )

        kwargs.setdefault(
            "attn_args",
            {
                "n_heads": attn_n_heads,
                "head_dim": attn_head_dim,
                "inter_channels": attn_gate_inter_channels,
            },
        )

        # input block
        block_cls = BLOCK_MAP[block_types[0]]
        blocks = [block_cls(dimensions, in_channels, n_channels, **kwargs)]
        in_c = n_channels
        for block_type, mult in zip(block_types[1:-1], channel_mults):
            out_c = int(in_c * mult)
            block_cls = BLOCK_MAP[block_type]
            block = block_cls(dimensions, in_c, out_c, **(kwargs | {"bias": True}))
            blocks.append(block)
            in_c = out_c
        # output block
        block_cls = BLOCK_MAP[block_types[-1]]
        blocks += [
            block_cls(dimensions, in_c, out_channels, **(kwargs | {"bias": True}))
        ]
        super().__init__(*blocks)

    @property
    def conv_layers(self) -> list[nn.Module]:
        """List of convolutional and pooling layers."""
        return [
            mod
            for mod in self.modules()
            if isinstance(mod, _ConvNd | _AvgPoolNd | _MaxPoolNd)
        ]

    @property
    def n_hidden(self) -> int:
        """Number of hidden blocks in the model."""
        return len(self.conv_layers) - 2

    def receptive_field(self) -> tuple[int, ...]:
        """Calculate the receptive field size of the discriminator."""
        layers = self.conv_layers
        n_dim = (
            len(layers[0].kernel_size)
            if isinstance(layers[0].kernel_size, tuple)
            else 1
        )
        r = [1] * n_dim
        for layer in reversed(layers):
            for i in range(n_dim):
                k = (
                    layer.kernel_size[i]
                    if isinstance(layer.kernel_size, tuple)
                    else layer.kernel_size
                )
                if hasattr(layer, "dilation"):
                    d = (
                        layer.dilation[i]
                        if isinstance(layer.dilation, tuple)
                        else layer.dilation
                    )
                    k = d * (k - 1) + 1
                s = layer.stride[i] if isinstance(layer.stride, tuple) else layer.stride
                p = (
                    layer.padding[i]
                    if isinstance(layer.padding, tuple)
                    else layer.padding
                )
                if isinstance(p, str):
                    p = 0
                r[i] = (r[i] - p) * s + k
        return tuple(r)


class PixelDiscriminator(BlockDiscriminator):
    """PixelGAN or 1x1-PatchGAN discriminator."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        n_hidden: int = 1,
        block_types: tuple[str, ...] | None = None,
        **kwargs,
    ):
        """Constructs a sequential model of 3 blocks, no downsampling, no padding.

        Args:
          dimensions: Number of dimensions.
          in_channels: Number of input channels.
          n_channels: Number of channels in the first layer.
          n_hidden: Number of hidden blocks in the model;
            only takes effect if block_types is None.
          block_types: Types of down blocks.
          channel_mults: Channel multipliers for each block.
          out_channels: Output channels (should generally be 1, i.e. true/fake).
          attn_n_heads: Number of attention heads.
          attn_head_dim: Dimension of the attention head.
          attn_gate_inter_channels: Number of intermediate channels for the attention gate.
          kwargs: Additional arguments for the blocks.
        """
        if block_types is None:
            block_types = ("ConvBlock", "ActConvBlock") + (
                "NormActConvBlock",
            ) * n_hidden
        kwargs["block_types"] = block_types
        kwargs["kernel_size"] = 1
        kwargs["stride"] = 1
        kwargs["padding"] = 0
        super().__init__(dimensions, in_channels, n_channels, **kwargs)


class PatchDiscriminator(BlockDiscriminator):
    """PatchGAN discriminator."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        n_hidden: int = 3,
        block_types: tuple[str, ...] | None = None,
        **kwargs,
    ):
        """Constructs a sequential model of 5 PatchGAN blocks.

        Args:
          dimensions: Number of dimensions.
          in_channels: Number of input channels.
          n_channels: Number of channels in the first layer.
          n_hidden: Number of hidden blocks in the model;
            only takes effect if block_types is None.
          block_types: Types of down blocks.
          channel_mults: Channel multipliers for each block.
          out_channels: Output channels (should generally be 1, i.e. true/fake).
          attn_n_heads: Number of attention heads.
          attn_head_dim: Dimension of the attention head.
          attn_gate_inter_channels: Number of intermediate channels for the attention gate.
          kwargs: Additional arguments for the blocks.
        """
        if block_types is None:
            block_types = (
                ("ConvDownBlock", "ActConvDownBlock")
                + ("NormActConvDownBlock",) * (n_hidden - 2)
                + (
                    "NormActConvBlock",
                    "NormActConvBlock",
                )
            )
        kwargs["block_types"] = block_types
        super().__init__(dimensions, in_channels, n_channels, **kwargs)


class AttnPatchDiscriminator(BlockDiscriminator):
    """PatchGAN discriminator with attention layers in the hidden blocks."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        n_hidden: int = 3,
        block_types: tuple[str, ...] | None = None,
        **kwargs,
    ):
        """Constructs a sequential model of 5 PatchGAN blocks.

        Args:
          dimensions: Number of dimensions.
          in_channels: Number of input channels.
          n_channels: Number of channels in the first layer.
          n_hidden: Number of hidden blocks in the model;
            only takes effect if block_types is None.
          block_types: Types of down blocks.
          channel_mults: Channel multipliers for each block.
          out_channels: Output channels (should generally be 1, i.e. true/fake).
          attn_n_heads: Number of attention heads.
          attn_head_dim: Dimension of the attention head.
          attn_gate_inter_channels: Number of intermediate channels for the attention gate.
          kwargs: Additional arguments for the blocks.
        """
        if block_types is None:
            block_types = (
                ("ConvDownBlock", "ActConvDownBlock")
                + ("NormActAttnConvDownBlock",) * (n_hidden - 2)
                + (
                    "NormActAttnConvBlock",
                    "NormActConvBlock",
                )
            )
        kwargs["block_types"] = block_types
        super().__init__(dimensions, in_channels, n_channels, **kwargs)


class AntialiasingDiscriminator(BlockDiscriminator):
    """Discriminator with separated feature mapping and downsampling block pairs."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        patch_size: int = 32,
        n_hidden: int = 3,
        block_types: tuple[str, ...] | None = None,
        channel_mults: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """Constructs a discriminator of 5 antialiasing block pairs.

        Args:
          dimensions: Number of dimensions.
          in_channels: Number of input channels.
          n_channels: Number of channels in the first layer.
          patch_size: Size of the patch
          n_hidden: Number of hidden block pairs in the model;
            only takes effect if block_types is None.
          block_types: Types of down blocks.
          channel_mults: Channel multipliers for each block.
          out_channels: Output channels (should generally be 1, i.e. true/fake).
          attn_n_heads: Number of attention heads.
          attn_head_dim: Dimension of the attention head.
          attn_gate_inter_channels: Number of intermediate channels for the attention gate.
          kwargs: Additional arguments for the blocks.
        """
        self.patch_size = patch_size
        if block_types is None:
            block_types = (
                ("ConvBlock", "ActConvDownsampleBlock", "ConvBlock")
                + ("NormActConvDownsampleBlock", "ConvBlock") * (n_hidden - 1)
                + ("NormActConvBlock",)
            )
            channel_mults = (
                1,
                2,
            ) + (
                1,
                2,
            ) * (n_hidden - 1)
        kwargs["block_types"] = block_types
        kwargs["channel_mults"] = channel_mults
        super().__init__(dimensions, in_channels, n_channels, **kwargs)

    @property
    def n_hidden(self) -> int:
        """Number of hidden block pairs in the model."""
        return len(self.conv_layers) // 2 - 1


class AntialiasingPatchDiscriminator(AntialiasingDiscriminator):
    """PatchGAN discriminator with separated feature mapping and downsampling block pairs."""

    def forward(self, x: torch.Tensor, patch_size: int | None = None) -> torch.Tensor:
        """Forward pass."""
        dim = len(x.shape)
        B, C, W = x.size(0), x.size(1), x.size(2)
        if patch_size is None:
            patch_size = self.patch_size
        if dim == 3:
            X = W // patch_size
            x = x.view(B, C, X, patch_size)
            x = x.permute(0, 2, 1, 3).contiguous().view(B * X, C, patch_size)
        elif dim == 4:
            H, W = x.size(2), x.size(3)
            X = H // patch_size
            Y = W // patch_size
            x = x.view(B, C, Y, patch_size, X, patch_size)
            x = (
                x.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(B * Y * X, C, patch_size, patch_size)
            )
        elif dim == 5:
            H, W, D = x.size(2), x.size(3), x.size(4)
            X = H // patch_size
            Y = W // patch_size
            Z = D // patch_size
            x = x.view(B, C, Z, patch_size, Y, patch_size, X, patch_size)
            x = (
                x.permute(0, 2, 4, 6, 1, 3, 5, 7)
                .contiguous()
                .view(B * Z * Y * X, C, patch_size, patch_size, patch_size)
            )
        return super().forward(x)

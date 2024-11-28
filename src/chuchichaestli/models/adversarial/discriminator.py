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

from torch import nn
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.adversarial.blocks import BLOCK_MAP


class NLayerDiscriminator(nn.Sequential):
    """A PatchGAN discriminator as in Pix2Pix.

    https://arxiv.org/abs/1611.07004
    """
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        block_types: tuple[str, ...] = (
            'ConvDownBlock',
            'ActConvDownBlock',
            'NormActConvDownBlock',
            'NormActConvBlock',
            'NormActConvBlock',
        ),
        channel_mults: tuple[int, ...] | None = None,
        out_channels: int = 1,
        **kwargs,
    ):
        """Construct a PatchGAN discriminator.

        Args:
          dimensions (int): Number of dimensions.
          in_channels (int): Number of input channels.
          n_channels (int): Number of channels in the first layer.
          block_types (tuple[str]): Types of down blocks.
          channel_mults (tuple[int]): Channel multipliers for each block.
          out_channels (int): Output channels (should generally be 1, i.e. true/fake).
          kwargs (dict): Additional arguments for the blocks.
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
        elif len(channel_mults) < n_blocks-2:
            raise ValueError(
                f"Not enough channel multipliers. Must be at least len(block_types)-2 = {n_blocks-2}."
            )

        # input block
        block_cls = BLOCK_MAP[block_types[0]]
        blocks = [block_cls(dimensions, in_channels, n_channels, **kwargs)]
        in_c = n_channels
        for block_type, mult in zip(block_types[1:-1], channel_mults):
            out_c = int(in_c * mult)
            block_cls = BLOCK_MAP[block_type]
            block = block_cls(dimensions, in_c, out_c, **kwargs | {"bias": False})
            blocks.append(block)
            in_c = out_c
        # output block
        block_cls = BLOCK_MAP[block_types[-1]]
        blocks += [block_cls(dimensions, in_c, out_channels, **kwargs)]
        super().__init__(*blocks)

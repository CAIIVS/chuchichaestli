"""Tests for the patchgan module.

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

import pytest
import torch
from chuchichaestli.models.adversarial.blocks import ConvDownBlock
from chuchichaestli.models.adversarial.discriminator import NLayerDiscriminator


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,act_fn,norm_type",
    [
        (1, 32, 64, "leakyrelu,0.2", "batch"),
        (2, 64, 128, "leakyrelu,0.2", "batch"),
        (3, 64, 128, "leakyrelu,0.2", "batch"),
    ]
)
def test_ConvDownBlock_forward_pass(
    dimensions,
    in_channels,
    out_channels,
    act_fn,
    norm_type,
    num_groups = 32,
    img_wh = 32,
):
    """Test forward pass of a ConvDownBlock module."""
    block = ConvDownBlock(
        dimensions,
        in_channels,
        out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
        num_groups=num_groups
    )
    x_shape = (1, in_channels) + (img_wh,) * dimensions
    x = torch.randn(*x_shape)
    out = block(x)
    print(out.shape)


def test_ConvDownBlock_info(
    dimensions = 3,
    in_channels = 64,
    out_channels = 128,
    act_fn = "leakyrelu,0.2",
    norm_type = "batch",
    num_groups = 32,
    img_wh = 64,
):
    """Test print a torchinfo pass of a ConvDownBlock module."""
    block = ConvDownBlock(
        dimensions,
        in_channels,
        out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
        num_groups=num_groups
    )
    print("\n# ConvDownBlock")
    try:
        from torchinfo import summary
        summary(
            block,
            (4, in_channels) + (img_wh,)*dimensions,
            col_names=["input_size", "output_size", "num_params"]
        )
    except ImportError:
        print(block)
    print()


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,out_channels,block_types,channel_mults,act_fn,norm_type",
    [
        (1, 1, 64, 1, ('ConvDownBlock', 'ActConvDownBlock', *(('NormActConvDownBlock',)*3)),
         (2,)*3, "leakyrelu,0.2", "batch"),
        (2, 1, 64, 1, ('ConvDownBlock', 'ActConvDownBlock', *(('NormActConvDownBlock',)*3)),
         (2,)*3, "leakyrelu,0.2", "batch"),
    ]
)
def test_NLayerDiscriminator_forward_pass(
    dimensions,
    in_channels,
    n_channels,
    out_channels,
    block_types,
    channel_mults,
    act_fn,
    norm_type,
    img_wh = 128
):
    """Test forward pass of a NLayerDiscriminator."""
    block = NLayerDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        block_types=block_types,
        channel_mults=channel_mults,
        out_channels=out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
    )
    x_shape = (1, in_channels) + (img_wh,) * dimensions
    x = torch.randn(*x_shape)
    out = block(x)
    print(out.shape)


def test_NLayerDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    block_types=("ConvDownBlock", "ActConvDownBlock", "NormConvDownBlock", "NormConvBlock", "NormConvBlock"),
    channel_mults=(2,)*3,
    out_channels=1,
    act_fn="leakyrelu,0.2",
    norm_type="batch",
    img_wh = 128,
):
    """Test print a torchinfo pass of a NLayerDiscriminator module."""
    model = NLayerDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        
    )
    print("\n# TamingNLayerDiscriminator")
    try:
        from torchinfo import summary
        summary(model, (4, 1)+(img_wh,)*dimensions, col_names=["input_size", "output_size", "num_params"])
    except ImportError:
        print(model)
    print()

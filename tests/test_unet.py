"""Tests for the UNet model.

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

from chuchichaestli.models.unet import UNet

import pytest
import torch


@pytest.fixture
def unet_conv_2d():
    """Create an instance of the UNet model."""
    return UNet(
        dimensions=2,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
    )


def test_throws_error_on_invalid_dimension():
    """Test that the UNet model throws an error when an invalid dimension is passed."""
    with pytest.raises(ValueError):
        UNet(dimensions=4)


def test_throws_error_on_mismatched_lengths():
    """Test that the UNet model throws an error when the down and up block types have different lengths."""
    with pytest.raises(ValueError):
        UNet(
            down_block_types=("DownBlock", "AttnDownBlock"),
            up_block_types=("UpBlock", "AttnUpBlock", "AttnUpBlock"),
        )


def test_throws_error_on_mismatched_lengths_2():
    """Test that the UNet model throws an error when the down block types and out channels have different lengths."""
    with pytest.raises(ValueError):
        UNet(
            down_block_types=("DownBlock", "AttnDownBlock"),
            up_block_types=("UpBlock", "AttnUpBlock"),
            block_out_channels=(224, 448, 672),
        )


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,block_out_channels",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        # Attention test cases in 2D
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        (2, ("DownBlock", "DownBlock"), ("AttnUpBlock", "UpBlock"), (32, 64)),
        (2, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), (32, 64)),
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), (32, 64)),
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), (64, 128)),
        # Attention test cases in 3D
        (3, ("AttnDownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        (3, ("DownBlock", "DownBlock"), ("AttnUpBlock", "UpBlock"), (32, 64)),
        (3, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), (32, 64)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), (32, 64)),
        (3, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), (16, 32)),
    ],
)
def test_forward_pass(dimensions, down_block_types, up_block_types, block_out_channels):
    """Test the forward pass of the UNet model."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        resnet_groups=16,
    )
    input_dims = (1, 1) + (64,) * dimensions
    sample = torch.randn(*input_dims)  # Example input

    timestep = 0.5  # Example timestep
    output = model(sample, timestep)
    assert output.shape == input_dims  # Check output shape


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,block_out_channels",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (8, 16)),
        (2, ("DownBlock", "AttnDownBlock"), ("AttnUpBlock", "UpBlock"), (8, 16)),
        (3, ("DownBlock", "AttnDownBlock"), ("AttnUpBlock", "UpBlock"), (8, 16)),
    ],
)
def test_no_timestep(dimensions, down_block_types, up_block_types, block_out_channels):
    """Test the forward pass of the UNet model without a timestep."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        add_time_embedding=False,
        resnet_groups=8,
    )
    input_dims = (1, 1) + (64,) * dimensions
    sample = torch.randn(*input_dims)  # Example input

    output = model(sample)
    assert output.shape == input_dims

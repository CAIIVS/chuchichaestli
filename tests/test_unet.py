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
            block_out_channel_mults=(1, 2, 3),
        )


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        # Attention test cases in 2D
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("AttnUpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        # Attention test cases in 3D
        (3, ("AttnDownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("AttnUpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (3, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (
            2,
            ("DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2),
        ),
        (
            2,
            ("DownBlock", "DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2, 4),
        ),
        (
            3,
            ("DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2),
        ),
        (
            3,
            ("DownBlock", "DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2, 4),
        ),
        # AttentionGate test cases
        (
            2,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
        ),
        (
            3,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
        ),
    ],
)
def test_forward_pass(
    dimensions, down_block_types, up_block_types, n_channels, block_out_channel_mults
):
    """Test the forward pass of the UNet model."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channel_mults=block_out_channel_mults,
        n_channels=n_channels,
        res_groups=4,
        num_layers_per_block=1,
    )
    input_dims = (1, 1) + (64,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    timestep = 0.5  # Example timestep
    output = model(sample, timestep)
    assert output.shape == input_dims  # Check output shape


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 8, (1, 2)),
        (2, ("DownBlock", "AttnDownBlock"), ("AttnUpBlock", "UpBlock"), 8, (1, 2)),
        (3, ("DownBlock", "AttnDownBlock"), ("AttnUpBlock", "UpBlock"), 8, (1, 2)),
    ],
)
def test_no_timestep(
    dimensions, down_block_types, up_block_types, n_channels, block_out_channel_mults
):
    """Test the forward pass of the UNet model without a timestep."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        time_embedding=False,
        res_groups=8,
    )
    input_dims = (1, 1) + (64,) * dimensions
    sample = torch.randn(*input_dims)  # Example input

    output = model(sample)
    assert output.shape == input_dims


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults, in_channels, out_channels",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (2, 4), 1, 3),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (2, 4), 1, 3),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (2, 4), 1, 3),
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 16, (2, 4), 6, 3),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 16, (2, 4), 6, 3),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 16, (2, 4), 6, 3),
    ],
)
def test_out_channels(
    dimensions,
    down_block_types,
    up_block_types,
    n_channels,
    block_out_channel_mults,
    in_channels,
    out_channels,
):
    """Test the forward pass of the UNet model with a different number of output channels."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        in_channels=in_channels,
        out_channels=out_channels,
        res_groups=16,
    )
    input_dims = (1, in_channels) + (64,) * dimensions
    sample = torch.randn(*input_dims)
    timestep = 0.5
    output = model(sample, timestep)
    assert output.shape == (1, out_channels) + (64,) * dimensions


@pytest.mark.parametrize(
    "in_kernel_size,out_kernel_size,res_kernel_size",
    [
        (1, 1, 1),
        (3, 3, 3),
        (5, 5, 5),
        (7, 7, 7),
        (9, 9, 9),
        (1, 3, 5),
        (5, 3, 1),
        (3, 1, 5),
        (5, 1, 3),
        (1, 5, 3),
    ],
)
def test_kernel_sizes(in_kernel_size, out_kernel_size, res_kernel_size):
    """Test the forward pass of the UNet model with different kernel sizes."""
    model = UNet(
        dimensions=2,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
        n_channels=16,
        block_out_channel_mults=(1, 2),
        in_kernel_size=in_kernel_size,
        out_kernel_size=out_kernel_size,
        res_kernel_size=res_kernel_size,
        res_groups=16,
    )
    input_dims = (1, 1, 64, 64)
    sample = torch.randn(*input_dims)
    timestep = 0.5
    output = model(sample, timestep)
    assert output.shape == input_dims

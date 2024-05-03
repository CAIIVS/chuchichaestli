"""Tests for the UNet model."""

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
    ],
)
def test_forward_pass(dimensions, down_block_types, up_block_types, block_out_channels):
    """Test the forward pass of the UNet model."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
    )
    input_dims = (1, 1) + (64,) * dimensions
    sample = torch.randn(*input_dims)  # Example input

    timestep = 0.5  # Example timestep
    output = model(sample, timestep)
    assert output.shape == input_dims  # Check output shape

"""Tests for the UNet model."""

from chuchichaestli.models.unet import UNet

import pytest


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

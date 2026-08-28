# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the Upsampling module."""

import pytest
import torch
from chuchichaestli.models.downsampling import DownsampleUnshuffle
from chuchichaestli.models.upsampling import (
    Upsample,
    UpsampleInterpolate,
    UpsampleShuffle,
)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_forward(dimensions):
    """Test the forward method of the `Upsample` module."""
    # Create dummy input tensor
    input_shape = (1, 16) + (32,) * dimensions
    output_shape = (1, 16) + (64,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = Upsample(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_interpolate_forward(dimensions):
    """Test the forward method of the `UpsampleInterpolate` module."""
    # Create dummy input tensor
    input_shape = (1, 16) + (32,) * dimensions
    output_shape = (1, 16) + (64,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = UpsampleInterpolate(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_forward_with_large_batch_3d(dimensions):
    """Test the forward method of the upsample module with a large batch size."""
    # Create dummy input tensor
    input_shape = (128, 16) + (32,) * dimensions
    output_shape = (128, 16) + (64,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = Upsample(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("factor", [2, 4])
def test_upsampleshuffle_forward(dimensions, factor):
    """Test the forward method of the `UpsampleShuffle` module at every rank."""
    in_channels, out_channels = 32, 16
    input_shape = (2, in_channels) + (8,) * dimensions
    output_shape = (2, out_channels) + (8 * factor,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = UpsampleShuffle(
        dimensions=dimensions,
        in_channels=in_channels,
        out_channels=out_channels,
        factor=factor,
    )

    assert upsample.forward(input_tensor, None).shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_shuffle_samplers_are_inverse(dimensions):
    """Test that a shuffle sampler restores the shape an unshuffle sampler produced."""
    input_tensor = torch.randn(2, 16, *(16,) * dimensions)
    down = DownsampleUnshuffle(dimensions, 16, 32)
    latent = down(input_tensor)
    assert UpsampleShuffle(dimensions, 32, 16)(latent).shape == input_tensor.shape


def test_upsampleshuffle_throws_error_on_indivisible_channels():
    """Test that a channel count the factor cannot divide is rejected."""
    with pytest.raises(ValueError, match="Cannot shuffle"):
        UpsampleShuffle(dimensions=2, in_channels=7, out_channels=3)

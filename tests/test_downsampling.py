# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the downsampling module."""

import pytest
import torch
from chuchichaestli.models.downsampling import (
    Downsample,
    MaxPool,
    AdaptiveMaxPool,
    DownsampleUnshuffle,
    DownsampleInterpolate,
)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_forward(dimensions):
    """Test the forward method of the `Downsample` module."""
    # Create dummy input tensor
    input_shape = (1, 16) + (32,) * dimensions
    output_shape = (1, 16) + (16,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = Downsample(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_interpolate_forward(dimensions):
    """Test the forward method of the `DownsampleInterpolate` module."""
    # Create dummy input tensor
    input_shape = (1, 16) + (32,) * dimensions
    output_shape = (1, 16) + (16,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = DownsampleInterpolate(dimensions=dimensions, num_channels=None)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_forward_with_large_batch(dimensions):
    """Test the forward method of the downsample module with a large batch size."""
    # Create dummy input tensor
    input_shape = (128, 16) + (32,) * dimensions
    output_shape = (128, 16) + (16,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = Downsample(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_maxpool_forward(dimensions):
    """Test the forward method of the `MaxPool` module."""
    # Create dummy input tensor
    input_shape = (128, 16) + (32,) * dimensions
    output_shape = (128, 16) + (16,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = MaxPool(dimensions=dimensions)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_adamaxpool_forward(dimensions):
    """Test the forward method of the `AdaptiveMaxPool` module."""
    # Create dummy input tensor
    out_wh = (8,) * dimensions
    input_shape = (128, 16) + (32,) * dimensions
    output_shape = (128, 16) + out_wh
    input_tensor = torch.randn(input_shape)

    upsample = AdaptiveMaxPool(dimensions=dimensions, output_size=out_wh)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("factor", [2, 4])
def test_downsampleunshuffle_forward(dimensions, factor):
    """Test the forward method of the `DownsampleUnshuffle` module at every rank."""
    in_channels = 16
    out_channels = 2 * factor**dimensions
    input_shape = (2, in_channels) + (16,) * dimensions
    output_shape = (2, out_channels) + (16 // factor,) * dimensions
    input_tensor = torch.randn(input_shape)

    downsample = DownsampleUnshuffle(
        dimensions=dimensions,
        in_channels=in_channels,
        out_channels=out_channels,
        factor=factor,
    )

    assert downsample.forward(input_tensor, None).shape == output_shape


def test_downsampleunshuffle_throws_error_on_indivisible_channels():
    """Test that a channel count the factor cannot divide is rejected."""
    with pytest.raises(ValueError, match="Cannot unshuffle"):
        DownsampleUnshuffle(dimensions=2, in_channels=16, out_channels=6)

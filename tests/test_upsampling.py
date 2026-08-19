# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the Upsampling module."""

import pytest
import torch
from chuchichaestli.models.upsampling import Upsample, UpsampleInterpolate


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

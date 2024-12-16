"""Tests for the conv_attention module.

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
from chuchichaestli.models.attention.conv_attention import ConvAttention


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_conv_attention_forward(dimensions):
    """Test the ConvAttention module."""
    # Create an instance of ConvAttention
    n_channels = 32
    attention = ConvAttention(dimensions, n_channels)

    # Create a random input tensor
    batch_size = 2
    shape = (batch_size, n_channels) + (32,) * dimensions
    x = torch.randn(shape)

    # Perform forward pass
    output = attention(x, None)

    # Check output shape
    assert output.shape == x.shape

    # Check if the output tensor is on the same device as the input tensor
    assert output.device == x.device

    # Check if the output tensor is finite
    assert torch.isfinite(output).all()

    # Check if the output tensor is not NaN
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("dimensions,n_channels,img_wh", [(2, 32, 256), (2, 64, 128), (2, 512, 128)])
def test_conv_attention_different_sizes(dimensions, n_channels, img_wh):
    """Test the ConvAttention module."""
    # Create an instance of ConvAttention
    attention = ConvAttention(dimensions, n_channels)

    # Create a random input tensor
    batch_size = 2
    shape = (batch_size, n_channels) + (img_wh,) * dimensions
    x = torch.randn(shape)

    # Perform forward pass
    output = attention(x, None)

    # Check output shape
    assert output.shape == x.shape

    # Check if the output tensor is on the same device as the input tensor
    assert output.device == x.device

    # Check if the output tensor is finite
    assert torch.isfinite(output).all()

    # Check if the output tensor is not NaN
    assert not torch.isnan(output).any()

"""Tests for the downsampling modules.

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
from chuchichaestli.models.downsampling import Downsample1D, Downsample2D, Downsample3D


@pytest.fixture
def downsample1d():
    """Create an instance of the downsample1d module."""
    return Downsample1D(channels=16)


def test_forward_downsample1d(downsample1d):
    """Test the forward method of the downsample1d module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32)

    # Call the forward method
    output_tensor = downsample1d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 16)


@pytest.fixture
def downsample2d():
    """Create an instance of the downsample2d module."""
    return Downsample2D(channels=16)


def test_forward_downsample2d(downsample2d):
    """Test the forward method of the downsample2d module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32)

    # Call the forward method
    output_tensor = downsample2d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 16, 16)


@pytest.fixture
def downsample3d():
    """Create an instance of the downsample3d module."""
    return Downsample3D(channels=16)


def test_forward_downsample3d(downsample3d):
    """Test the forward method of the downsample3d module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32, 32)

    # Call the forward method
    output_tensor = downsample3d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 16, 16, 16)

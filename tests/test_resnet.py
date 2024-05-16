"""Tests for the ResnetBlock1D, ResnetBlock2D, and ResnetBlock3D modules.

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
from chuchichaestli.models.resnet import ResnetBlock1D, ResnetBlock2D, ResnetBlock3D


@pytest.fixture
def resnet_block1d():
    """Create an instance of the ResnetBlock1D module."""
    return ResnetBlock1D(in_channels=16, mid_channels=32, out_channels=16)


def test_forward_resnet_block1d(resnet_block1d):
    """Test the forward method of the ResnetBlock1D module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32)

    # Call the forward method
    output_tensor = resnet_block1d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 32)


@pytest.fixture
def resnet_block2d():
    """Create an instance of the ResnetBlock2D module."""
    return ResnetBlock2D(in_channels=64, out_channels=32)


def test_forward_resnet_block2d(resnet_block2d):
    """Test the forward method of the ResnetBlock2D module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 64, 32, 32)
    temb = torch.randn(1, 512)

    # Call the forward method
    output_tensor = resnet_block2d.forward(input_tensor, temb)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 32, 32, 32)


def test_forward_resnet_block2d_with_temb():
    """Test the forward method of the ResnetBlock2D module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 64, 32, 32)
    temb = torch.randn(1, 512)

    # Call the forward method
    output_tensor = ResnetBlock2D(in_channels=64, out_channels=32).forward(
        input_tensor, temb
    )

    # Check the output tensor shape
    assert output_tensor.shape == (1, 32, 32, 32)


def test_forward_resnet_block2d_channels_not_divisible():
    """Test the forward method of the ResnetBlock2D module."""
    # Call the forward method
    with pytest.raises(ValueError):
        ResnetBlock2D(in_channels=16, out_channels=32)


@pytest.fixture
def resnet_block3d():
    """Create an instance of the ResnetBlock3D module."""
    return ResnetBlock3D(in_channels=64, out_channels=32)


def test_forward_resnet_block3d(resnet_block3d):
    """Test the forward method of the ResnetBlock3D module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 64, 32, 32, 32)
    temb = torch.randn(1, 512)

    # Call the forward method
    output_tensor = resnet_block3d.forward(input_tensor, temb)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 32, 32, 32, 32)


def test_forward_resnet_block3d_channels_not_divisible():
    """Test the forward method of the ResnetBlock3D module."""
    # Call the forward method
    with pytest.raises(ValueError):
        ResnetBlock3D(in_channels=16, out_channels=32)

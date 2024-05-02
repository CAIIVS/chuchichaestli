"""Tests for the downsampling modules."""

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

"""Test the upsampling module."""

import pytest
import torch
from chuchichaestli.models.upsampling import Upsample1D, Upsample2D, Upsample3D


@pytest.fixture
def upsample3d():
    """Create an instance of the upsample module."""
    return Upsample3D(channels=16)


def test_forward(upsample3d):
    """Test the forward method of the upsample module.

    Args:
        upsample3d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32, 32)

    # Call the forward method
    output_tensor = upsample3d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64, 64)


def test_forward_with_large_batch_3d(upsample3d):
    """Test the forward method of the upsample module with a large batch size."""
    # Create dummy input tensor
    input_tensor = torch.randn(128, 16, 32, 32, 32)

    # Call the forward method
    output_tensor = upsample3d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (128, 16, 64, 64, 64)


def test_forward_with_output_size_3d(upsample3d):
    """Test the forward method of the upsample module with a specified output size.

    Args:
        upsample3d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32, 32)

    # Call the forward method with output_size
    output_tensor = upsample3d.forward(input_tensor, output_size=(128, 128, 128))

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 128, 128, 128)


def test_forward_with_norm_3d(upsample3d):
    """Test the forward method of the upsample module with normalization.

    Args:
        upsample3d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32, 32)

    # Call the forward method with normalization
    output_tensor = upsample3d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64, 64)


def test_forward_with_conv_3d(upsample3d):
    """Test the forward method of the upsample module with convolution.

    Args:
        upsample3d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32, 32)

    # Call the forward method with convolution
    output_tensor = upsample3d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64, 64)


def test_forward_with_conv_transpose_3d(upsample3d):
    """Test the forward method of the upsample module with convolution transpose.

    Args:
        upsample3d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32, 32)

    # Call the forward method with convolution transpose
    output_tensor = upsample3d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64, 64)


@pytest.fixture
def upsample1d():
    """Create an instance of the upsample module."""
    return Upsample1D(channels=16)


def test_forward_1d(upsample1d):
    """Test the forward method of the upsample module."""
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32)

    # Call the forward method
    output_tensor = upsample1d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64)


def test_forward_with_output_size_1d(upsample1d):
    """Test the forward method of the upsample module with a specified output size.

    Args:
        upsample1d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32)

    # Call the forward method with output_size
    output_tensor = upsample1d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64)


def test_forward_with_conv_1d(upsample1d):
    """Test the forward method of the upsample module with convolution.

    Args:
        upsample1d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32)

    # Call the forward method with convolution
    output_tensor = upsample1d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64)


def test_forward_with_conv_transpose_1d(upsample1d):
    """Test the forward method of the upsample module with convolution transpose.

    Args:
        upsample1d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32)

    # Call the forward method with convolution transpose
    output_tensor = upsample1d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64)


@pytest.fixture
def upsample2d():
    """Create an instance of the upsample module."""
    return Upsample2D(channels=16)


def test_forward_2d(upsample2d):
    """Test the forward method of the upsample module.

    Args:
        upsample2d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32)

    # Call the forward method
    output_tensor = upsample2d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64)


def test_forward_with_large_batch_2d(upsample2d):
    """Test the forward method of the upsample module with a large batch size."""
    # Create dummy input tensor
    input_tensor = torch.randn(128, 16, 32, 32)

    # Call the forward method
    output_tensor = upsample2d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (128, 16, 64, 64)


def test_forward_with_output_size_2d(upsample2d):
    """Test the forward method of the upsample module with a specified output size.

    Args:
        upsample2d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32)

    # Call the forward method with output_size
    output_tensor = upsample2d.forward(input_tensor, output_size=(128, 128))

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 128, 128)


def test_forward_with_norm_2d(upsample2d):
    """Test the forward method of the upsample module with normalization.

    Args:
        upsample2d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32)

    # Call the forward method with normalization
    output_tensor = upsample2d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64)


def test_forward_with_conv_2d(upsample2d):
    """Test the forward method of the upsample module with convolution.

    Args:
        upsample2d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32)

    # Call the forward method with convolution
    output_tensor = upsample2d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64)


def test_forward_with_conv_transpose_2d(upsample2d):
    """Test the forward method of the upsample module with convolution transpose.

    Args:
        upsample2d: The upsample module to be tested.

    Returns:
        None
    """
    # Create dummy input tensor
    input_tensor = torch.randn(1, 16, 32, 32)

    # Call the forward method with convolution transpose
    output_tensor = upsample2d.forward(input_tensor)

    # Check the output tensor shape
    assert output_tensor.shape == (1, 16, 64, 64)

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the attention module."""

import pytest
import torch
from chuchichaestli.models.attention.self_attention import SelfAttention


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_self_attention_1d(dimensions):
    """Test the SelfAttention1D module."""
    # Create an instance of SelfAttention1D
    n_channels = 64
    n_heads = 4
    head_dim = 32
    attention = SelfAttention(dimensions, n_channels, n_heads, head_dim)

    # Create a random input tensor
    batch_size = 8
    shape = (batch_size, n_channels) + (64,) * dimensions
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


@pytest.mark.parametrize(
    "dimensions,n_channels,img_wh", [(2, 512, 128), (2, 64, 128), (2, 32, 512)]
)
def test_self_attention_different_sizes(dimensions, n_channels, img_wh):
    """Test the SelfAttention module."""
    # Create an instance of Attention
    attention = SelfAttention(dimensions, n_channels, n_heads=1)

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

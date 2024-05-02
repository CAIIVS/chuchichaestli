"""Tests for the attention module."""

import torch
from chuchichaestli.models.attention import Attention, SelfAttention1D


def test_self_attention_1d():
    """Test the SelfAttention1D module."""
    # Create an instance of SelfAttention1D
    in_channels = 64
    n_head = 4
    dropout_rate = 0.2
    attention = SelfAttention1D(in_channels, n_head, dropout_rate)

    # Create a random input tensor
    batch_size = 8
    seq_length = 16
    x = torch.randn(batch_size, in_channels, seq_length)

    # Perform forward pass
    output = attention(x)

    # Check output shape
    assert output.shape == (batch_size, in_channels, seq_length)

    # Check if the output tensor is on the same device as the input tensor
    assert output.device == x.device

    # Check if the output tensor is finite
    assert torch.isfinite(output).all()

    # Check if the output tensor is not NaN
    assert not torch.isnan(output).any()


def test_attention():
    """Test the Attention module."""
    # Create an instance of Attention
    query_dim = 64
    heads = 8
    dim_head = 64
    dropout = 0.2
    bias = False
    rescale_output_factor = 1.0
    residual_connection = False
    attention = Attention(
        query_dim=query_dim,
        cross_attention_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
        bias=bias,
        rescale_output_factor=rescale_output_factor,
        residual_connection=residual_connection,
    )

    # Create a random input tensor
    batch_size = 8
    sequence_length = 16
    hidden_states = torch.randn(batch_size, sequence_length, query_dim)

    # Perform forward pass
    output = attention(hidden_states)

    # Check output shape
    assert output.shape == (batch_size, sequence_length, query_dim)

    # Check if the output tensor is on the same device as the input tensor
    assert output.device == hidden_states.device

    # Check if the output tensor is finite
    assert torch.isfinite(output).all()

    # Check if the output tensor is not NaN
    assert not torch.isnan(output).any()

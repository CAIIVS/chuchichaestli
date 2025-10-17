# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the attention gate module."""

import torch
import pytest

from chuchichaestli.models.attention.attention_gate import AttentionGate


@pytest.mark.parametrize(
    "dimension, feats", [(2, 64), (3, 64), (2, 32), (3, 16), (2, 128)]
)
def test_attention_gate_forward(dimension: int, feats: int):
    """Test the forward pass of the attention gate module."""
    # Create input tensors
    x_shape = (1, 32) + (64,) * dimension
    g_shape = (1, 64) + (feats,) * dimension
    x = torch.randn(x_shape)  # Example input tensor
    g = torch.randn(g_shape)  # Example guidance tensor

    # Create attention gate module
    attention_gate = AttentionGate(
        dimension=dimension, num_channels_g=64, num_channels_x=32, num_channels_inter=3
    )

    # Perform forward pass
    output = attention_gate.forward(x, g)

    # Check output shape
    assert output.shape == x.shape

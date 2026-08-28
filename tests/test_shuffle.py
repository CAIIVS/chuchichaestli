# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the dimension-generic pixel shuffling modules."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.shuffle import PixelShuffleND, PixelUnshuffleND


@pytest.mark.parametrize("factor", [2, 3])
def test_unshuffle_matches_torch_in_2d(factor):
    """Test that unshuffling reproduces `nn.PixelUnshuffle` exactly for 2D inputs."""
    x = torch.randn(2, 3, 6 * factor, 6 * factor)
    assert torch.equal(PixelUnshuffleND(2, factor)(x), nn.PixelUnshuffle(factor)(x))


@pytest.mark.parametrize("factor", [2, 3])
def test_shuffle_matches_torch_in_2d(factor):
    """Test that shuffling reproduces `nn.PixelShuffle` exactly for 2D inputs."""
    x = torch.randn(2, 3 * factor**2, 6, 6)
    assert torch.equal(PixelShuffleND(2, factor)(x), nn.PixelShuffle(factor)(x))


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("factor", [2, 3])
def test_unshuffle_moves_every_spatial_axis_into_the_channels(dimensions, factor):
    """Test that every spatial axis shrinks and the channels grow accordingly."""
    x = torch.randn(2, 3, *(6,) * dimensions)
    out = PixelUnshuffleND(dimensions, factor)(x)
    assert out.shape == (2, 3 * factor**dimensions, *(6 // factor,) * dimensions)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("factor", [2, 3])
def test_shuffle_inverts_unshuffle(dimensions, factor):
    """Test that shuffling restores exactly what unshuffling produced."""
    x = torch.randn(2, 3, *(6,) * dimensions)
    unshuffled = PixelUnshuffleND(dimensions, factor)(x)
    assert torch.equal(PixelShuffleND(dimensions, factor)(unshuffled), x)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_gradients_reach_the_input(dimensions):
    """Test that both modules are differentiable."""
    x = torch.randn(2, 3, *(8,) * dimensions, requires_grad=True)
    PixelShuffleND(dimensions)(PixelUnshuffleND(dimensions)(x)).sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_layer_type_is_still_recognised_as_shuffling():
    """Test that the class names keep the `SHUFFLE` label of the inspection utility."""
    from chuchichaestli.utils import get_layer_type

    assert get_layer_type(PixelShuffleND(2)) == get_layer_type(nn.PixelShuffle(2))
    assert get_layer_type(PixelUnshuffleND(2)) == get_layer_type(nn.PixelUnshuffle(2))

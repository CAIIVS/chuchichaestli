# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the attention gate module."""

import torch
import pytest

from chuchichaestli.models.attention.attention_gate import AttentionGate
from chuchichaestli.models.norm import Norm


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


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("alpha, expected", [(50.0, 3.0), (-50.0, 0.0)])
def test_attention_gate_prunes_x_not_g(dimension: int, alpha: float, expected: float):
    """Test that the attention coefficients are applied to x, not to the gate.

    The gate is pinned to pass everything through (alpha -> 1) or to prune
    everything (alpha -> 0), with the output transform set to the identity. What
    comes out is then the gated tensor itself, which tells x and g apart.
    """
    channels = 4
    gate = AttentionGate(
        dimension=dimension,
        num_channels_x=channels,
        num_channels_g=channels,
        num_channels_inter=2,
    )
    with torch.no_grad():
        gate.psi.weight.zero_()
        gate.psi.bias.fill_(alpha)
        gate.W_out.weight.copy_(
            torch.eye(channels).reshape((channels, channels) + (1,) * dimension)
        )
        gate.W_out.bias.zero_()
    gate.eval()  # the output norm falls back on its (identity) initial statistics

    shape = (1, channels) + (8,) * dimension
    out = gate(torch.full(shape, 3.0), torch.full(shape, 7.0))
    assert torch.allclose(out, torch.full(shape, expected), atol=1e-4)


def _count_interpolations(monkeypatch) -> list[int]:
    """Patch the module's interpolate to tally how often it runs."""
    from chuchichaestli.models.attention import attention_gate as module

    calls = [0]
    original = module.F.interpolate

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", counted)
    return calls


def test_attention_gate_skips_resampling_on_a_shared_grid(monkeypatch):
    """Test that co-resolution inputs are gated without any grid resampling."""
    calls = _count_interpolations(monkeypatch)
    gate = AttentionGate(dimension=2, num_channels_x=8, num_channels_g=8)
    x = torch.randn(1, 8, 16, 16)
    out = gate(x, torch.randn(1, 8, 16, 16))
    assert out.shape == x.shape
    assert calls[0] == 0


@pytest.mark.parametrize("g_size", [8, 32])
def test_attention_gate_resamples_across_grids(monkeypatch, g_size: int):
    """Test that a gating signal on another grid is resampled onto the one of x."""
    calls = _count_interpolations(monkeypatch)
    gate = AttentionGate(dimension=2, num_channels_x=8, num_channels_g=8)
    x = torch.randn(1, 8, 16, 16)
    out = gate(x, torch.randn(1, 8, g_size, g_size))
    assert out.shape == x.shape
    assert calls[0] == 1  # the gating signal only; alpha is already on x's grid


@pytest.mark.parametrize("size", [16, 17])
def test_attention_gate_subsample_factor(monkeypatch, size: int):
    """Test that a subsample factor coarsens the grid but keeps the output shape."""
    calls = _count_interpolations(monkeypatch)
    gate = AttentionGate(
        dimension=2, num_channels_x=8, num_channels_g=8, subsample_factor=2
    )
    x = torch.randn(1, 8, size, size)
    out = gate(x, torch.randn(1, 8, size, size))
    assert out.shape == x.shape
    # the gating signal drops onto the coarser grid, alpha is lifted back off it
    assert calls[0] == 2
    assert gate.W_x(x).shape[2:] == (size // 2,) * 2


def test_attention_gate_normalizes_its_output_by_default():
    """Test that batch normalization follows the output transform, as in the reference."""
    gate = AttentionGate(dimension=2, num_channels_x=8, num_channels_g=8)
    assert isinstance(gate.norm_out, Norm)
    assert gate.norm_out.ntype == "batch"


def test_attention_gate_without_an_output_norm():
    """Test that the output normalization can be dropped."""
    gate = AttentionGate(
        dimension=2, num_channels_x=8, num_channels_g=8, out_norm_type=None
    )
    assert gate.norm_out is None
    assert not [key for key in gate.state_dict() if key.startswith("norm_out")]


@pytest.mark.parametrize("out_norm_type", ["batch", "instance", "group", None])
def test_attention_gate_output_norm(out_norm_type: str | None):
    """Test that a requested output normalization is built and applied."""
    gate = AttentionGate(
        dimension=2,
        num_channels_x=8,
        num_channels_g=8,
        out_norm_type=out_norm_type,
        out_norm_groups=2,
    )
    x = torch.randn(2, 8, 8, 8)
    assert gate(x, torch.randn(2, 8, 8, 8)).shape == x.shape


@pytest.mark.parametrize(
    "out_norm_type, message",
    [("batch", "more than 1 value per channel"), ("instance", "more than 1 spatial")],
)
def test_attention_gate_output_norm_needs_something_to_reduce_over(
    out_norm_type: str, message: str
):
    """Test the documented limit of the reference's batch normalization."""
    gate = AttentionGate(
        dimension=2,
        num_channels_x=8,
        num_channels_g=8,
        out_norm_type=out_norm_type,
    )
    gate.train()
    with pytest.raises(ValueError, match=message):
        gate(torch.randn(1, 8, 1, 1), torch.randn(1, 8, 1, 1))


@pytest.mark.parametrize("out_norm_type", [None, "group", "layer", "rms"])
def test_attention_gate_trains_over_a_unit_spatial_extent(out_norm_type: str | None):
    """Test which output norms train on one sample over a single pixel.

    Batch and instance normalization both need more than one element to reduce
    over, so neither is usable at that size; the rest are.
    """
    gate = AttentionGate(
        dimension=2,
        num_channels_x=8,
        num_channels_g=8,
        out_norm_type=out_norm_type,
        out_norm_groups=2,
    )
    gate.train()
    gate(torch.randn(1, 8, 1, 1), torch.randn(1, 8, 1, 1)).sum().backward()

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the normalization module."""

import pytest
import torch
from typing import get_args

from chuchichaestli.models.norm import AdaNorm, AdaptiveBatchNorm, Norm, NormTypes


def test_norm_keeps_the_wrapped_layer_defaults():
    """Test that `affine=None` leaves each layer's own affine default alone."""
    assert Norm(2, "group", 8, 4).norm.affine is True
    assert Norm(2, "instance", 8, 0).norm.affine is False
    assert Norm(2, "batch", 8, 0).norm.affine is True
    assert Norm(2, "layer", 8, 0).norm.elementwise_affine is True
    assert Norm(2, "rms", 8, 0).norm.elementwise_affine is True


@pytest.mark.parametrize("norm_type", ["group", "instance", "batch"])
def test_norm_drops_the_affine_parameters(norm_type):
    """Test that `affine=False` removes the learned scale and shift."""
    norm = Norm(2, norm_type, 8, 4, affine=False)
    assert norm.norm.affine is False
    assert norm.norm.weight is None


@pytest.mark.parametrize("norm_type", ["layer", "rms"])
def test_norm_drops_the_elementwise_affine_parameters(norm_type):
    """Test that `affine=False` maps to `elementwise_affine` where that is the name."""
    norm = Norm(2, norm_type, 8, 0, affine=False)
    assert norm.norm.elementwise_affine is False
    assert norm.norm.weight is None


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("norm_type", get_args(NormTypes))
def test_adanorm_preserves_the_input_shape(dimensions, norm_type):
    """Test the AdaNorm forward pass across dimensions and normalization types."""
    norm = AdaNorm(dimensions, norm_type, 8, 4, 16)
    x = torch.randn(3, 8, *(6,) * dimensions)
    emb = torch.randn(3, 16)
    assert norm(x, emb).shape == x.shape


@pytest.mark.parametrize("norm_type", ["group", "layer", "rms"])
def test_adanorm_starts_out_as_the_plain_normalization(norm_type):
    """Test that the zero-initialized projection leaves the normalization untouched."""
    norm = AdaNorm(2, norm_type, 8, 4, 16)
    x = torch.randn(3, 8, 6, 6)
    emb = torch.randn(3, 16)
    assert torch.allclose(norm(x, emb), norm.norm(x), atol=1e-6)


def test_adanorm_without_zero_init_modulates_from_the_start():
    """Test that `zero_init=False` leaves the projection randomly initialized."""
    norm = AdaNorm(2, "group", 8, 4, 16, zero_init=False)
    x = torch.randn(3, 8, 6, 6)
    emb = torch.randn(3, 16)
    assert not torch.allclose(norm(x, emb), norm.norm(x), atol=1e-6)


def test_adanorm_modulates_each_sample_separately():
    """Test that two embeddings scale and shift their own sample only."""
    torch.manual_seed(0)
    norm = AdaNorm(2, "group", 8, 4, 16)
    torch.nn.init.normal_(norm.proj.weight, std=0.5)
    x = torch.randn(3, 8, 6, 6)
    emb = torch.randn(3, 16)
    residual = norm(x, emb) - norm.norm(x)
    assert not torch.allclose(residual[0], residual[1], atol=1e-6)


def test_adanorm_gradients_reach_the_projection():
    """Test that the embedding path is trainable."""
    norm = AdaNorm(2, "group", 8, 4, 16)
    x = torch.randn(3, 8, 6, 6)
    emb = torch.randn(3, 16)
    norm(x, emb).sum().backward()
    assert norm.proj.weight.grad is not None
    assert norm.proj.bias.grad.abs().sum() > 0


def test_adaptive_batch_norm_starts_as_plain_batch_norm():
    """Test that the mixing parameters are initialized rather than left as raw memory."""
    norm = AdaptiveBatchNorm(2, 8)
    assert torch.isfinite(norm.a).all()
    assert torch.isfinite(norm.b).all()
    x = torch.randn(4, 8, 6, 6)
    assert torch.allclose(norm(x), norm.bn(x), atol=1e-6)


def test_adaptive_batch_norm_keeps_both_paths_trainable():
    """Test that the input and the normalized path both receive gradients.

    The loss is quadratic on purpose: a plain sum has no gradient towards `b`,
    since the normalized path is zero-mean per channel.
    """
    torch.manual_seed(0)
    norm = AdaptiveBatchNorm(2, 8)
    norm(torch.randn(4, 8, 6, 6)).pow(2).sum().backward()
    assert norm.a.grad.abs().sum() > 0
    assert norm.b.grad.abs().sum() > 0

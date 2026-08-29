# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the normalization module."""

import torch
from chuchichaestli.models.norm import AdaptiveBatchNorm


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

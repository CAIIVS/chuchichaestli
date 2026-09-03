# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the signal extension modes of the wavelet transform."""

import pytest
import torch

from chuchichaestli.dwt.modes import (
    MODE_TO_CODE,
    extension_indices,
    pad_signal,
)


# `[1, 2, 3, 4]` extended by three samples on either side, as PyWavelets pads it.
GOLDEN_PADS = {
    "zero": [0, 0, 0, 1, 2, 3, 4, 0, 0, 0],
    "constant": [1, 1, 1, 1, 2, 3, 4, 4, 4, 4],
    "symmetric": [3, 2, 1, 1, 2, 3, 4, 4, 3, 2],
    "reflect": [4, 3, 2, 1, 2, 3, 4, 3, 2, 1],
    "periodic": [2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
    "periodization": [2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
    "antisymmetric": [-3, -2, -1, 1, 2, 3, 4, -4, -3, -2],
    "antireflect": [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
}

MODES = sorted(MODE_TO_CODE)


class TestExtension:
    """Tests for the extension a mode produces."""

    @pytest.mark.parametrize("mode", MODES)
    def test_matches_the_published_extension(self, mode):
        """Test each mode against a hand-checked extension of a short ramp."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        out = pad_signal(x, 0, 3, 3, mode)
        assert out.tolist() == pytest.approx(GOLDEN_PADS[mode])

    @pytest.mark.parametrize("mode", MODES)
    def test_the_core_samples_are_untouched(self, mode):
        """Test that the original signal survives in the middle of the extension."""
        x = torch.randn(7, dtype=torch.float64)
        out = pad_signal(x, 0, 4, 2, mode)
        assert out.shape == (13,)
        assert torch.allclose(out[4:11], x)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("pad", [1, 5, 9, 23])
    def test_padding_may_exceed_the_signal_length(self, mode, pad):
        """Test extensions wider than the signal, which `F.pad` cannot produce."""
        x = torch.randn(4, dtype=torch.float64)
        out = pad_signal(x, 0, pad, pad, mode)
        assert out.shape == (4 + 2 * pad,)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("mode", MODES)
    def test_a_single_sample_signal_extends(self, mode):
        """Test the degenerate signal every folding rule has to cope with."""
        x = torch.tensor([2.0], dtype=torch.float64)
        out = pad_signal(x, 0, 2, 2, mode)
        assert out.shape == (5,)
        assert out[2].item() == 2.0

    @pytest.mark.parametrize("mode", MODES)
    def test_the_extension_is_linear(self, mode):
        """Test that every mode is a linear map, which is what makes it differentiable."""
        a, b = torch.randn(6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
        pa, pb, pab = (pad_signal(v, 0, 4, 4, mode) for v in (a, b, a + b))
        assert torch.allclose(pab, pa + pb, atol=1e-12)

    @pytest.mark.parametrize("mode", MODES)
    def test_gradients_flow_through_the_extension(self, mode):
        """Test that the gather backpropagates onto the source samples."""
        x = torch.randn(6, dtype=torch.float64, requires_grad=True)
        pad_signal(x, 0, 3, 3, mode).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_zero_padding_needs_no_extension(self):
        """Test that a request for no padding returns the input untouched."""
        x = torch.randn(5)
        assert pad_signal(x, 0, 0, 0, "symmetric") is x


class TestAxesAndShapes:
    """Tests for extending tensors of higher rank."""

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
    def test_only_the_requested_axis_grows(self, mode, axis):
        """Test that the extension applies to one axis of a rank-3 tensor."""
        x = torch.randn(3, 4, 5, dtype=torch.float64)
        out = pad_signal(x, axis, 2, 1, mode)
        expected = list(x.shape)
        expected[axis % 3] += 3
        assert list(out.shape) == expected

    @pytest.mark.parametrize("mode", MODES)
    def test_every_lane_is_extended_the_same_way(self, mode):
        """Test that the extension is applied independently per lane."""
        x = torch.randn(2, 6, dtype=torch.float64)
        out = pad_signal(x, 1, 3, 3, mode)
        for i in range(2):
            assert torch.allclose(out[i], pad_signal(x[i], 0, 3, 3, mode))


class TestExtensionIndices:
    """Tests for the cached gather the extension is built from."""

    @pytest.mark.parametrize("mode", MODES)
    def test_the_gather_covers_the_extended_length(self, mode):
        """Test the shape and dtype of the four returned tensors."""
        sign, index, lo, hi = extension_indices(5, 2, 3, mode)
        assert sign.shape == index.shape == lo.shape == hi.shape == (10,)
        assert index.dtype is torch.int64
        assert bool(((index >= 0) & (index < 5)).all())

    def test_the_gather_is_cached(self):
        """Test that repeated requests return the very same tensors."""
        first = extension_indices(8, 2, 2, "symmetric")
        assert extension_indices(8, 2, 2, "symmetric") is first

    def test_only_the_anchored_mode_uses_the_edge_terms(self):
        """Test that `antireflect` is the one mode needing the edge anchors."""
        for mode in MODES:
            _, _, lo, hi = extension_indices(6, 3, 3, mode)
            used = bool((lo != 0).any() or (hi != 0).any())
            assert used == (mode == "antireflect")

    def test_an_empty_signal_raises(self):
        """Test that a signal needs at least one sample."""
        with pytest.raises(ValueError, match="at least one sample"):
            extension_indices(0, 1, 1, "zero")

    def test_negative_padding_raises(self):
        """Test that the extension widths must not be negative."""
        with pytest.raises(ValueError, match="must not be negative"):
            extension_indices(4, -1, 1, "zero")

    def test_an_unknown_mode_raises(self):
        """Test that an unsupported extension mode is reported."""
        with pytest.raises(ValueError, match="Unsupported signal extension mode"):
            extension_indices(4, 1, 1, "nope")

    def test_the_mode_codes_are_unique(self):
        """Test the codes shared with the compiled implementation."""
        assert len(set(MODE_TO_CODE.values())) == len(MODE_TO_CODE)

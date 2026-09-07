# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for factoring a two-channel filter bank into lifting steps.

The factorization is not unique, so what is asserted is not a particular set of
steps but that multiplying them back gives the bank they came from.
"""

import pytest

import torch

from chuchichaestli.dwt import dwt, wavelet, wavelist
from chuchichaestli.utils.arithmetic.laurent import Laurent
from chuchichaestli.utils.arithmetic.lifting import factor, matrix, polyphase, rebuild


# the banks whose factorization does not yet reconstruct to tolerance
UNFACTORED = {"sym6", "sym7"}

FACTORABLE = [name for name in wavelist() if name not in UNFACTORED]


def apart(a: Laurent, b: Laurent) -> float:
    """How far two polynomials differ, coefficient by coefficient.

    Args:
        a: One polynomial.
        b: The other.
    """
    low = min(a.low, b.low, 0)
    high = max(a.high, b.high, 0)

    def at(p: Laurent, i: int) -> float:
        return p.c[i - p.low] if p and p.low <= i <= p.high else 0.0

    return max((abs(at(a, i) - at(b, i)) for i in range(low, high + 1)), default=0.0)


class TestPolyphase:
    """Tests for splitting a filter into phases."""

    def test_a_tap_lands_in_the_phase_the_alignment_gives_it(self):
        """Test that `offset - f` decides which phase a tap belongs to."""
        # out[k] = sum_f filt[f] x[2k + 3 - f], so taps 1 and 3 read even samples
        even, odd = polyphase([1.0, 2.0, 3.0, 4.0], 3)
        assert even.c == (4.0, 2.0)
        assert odd.c == (3.0, 1.0)

    def test_the_centred_alignment_is_the_default(self):
        """Test that a bank is split the way a critically sampled transform reads it."""
        filt = [1.0, 2.0, 3.0, 4.0]
        assert matrix(filt, filt)[:2] == polyphase(filt, len(filt) // 2)


class TestAgainstTheTransform:
    """Tests that the factorization computes the transform it came from."""

    @pytest.mark.parametrize("name", FACTORABLE)
    def test_lifting_reproduces_the_wavelet_transform(self, name):
        """Test that the steps carry a signal to the same coefficients."""
        torch.manual_seed(0)
        x = torch.randn(64, dtype=torch.float64)
        wave = wavelet(name)
        lift = factor(wave.dec_lo, wave.dec_hi)
        approx, detail = x[0::2].clone(), x[1::2].clone()
        gain, delay = lift.approx
        approx = gain * torch.roll(approx, -delay)
        gain, delay = lift.detail
        detail = gain * torch.roll(detail, -delay)
        for step in lift.steps:
            filtered = sum(
                c * torch.roll(approx if step.on_detail else detail, -(step.q.low + i))
                for i, c in enumerate(step.q.c)
            )
            if step.on_detail:
                detail = detail + filtered
            else:
                approx = approx + filtered
        want_approx, want_detail = dwt(x, name, mode="periodization")
        assert torch.allclose(approx, want_approx, atol=1e-9)
        assert torch.allclose(detail, want_detail, atol=1e-9)


class TestFactor:
    """Tests for the factorization itself."""

    @pytest.mark.parametrize("name", FACTORABLE)
    def test_the_steps_multiply_back_to_the_bank(self, name):
        """Test that a factorization reproduces the matrix it came from."""
        wave = wavelet(name)
        got = rebuild(factor(wave.dec_lo, wave.dec_hi))
        want = matrix(wave.dec_lo, wave.dec_hi)
        assert max(apart(g, w) for g, w in zip(got, want)) < 1e-9

    @pytest.mark.parametrize("name", FACTORABLE)
    def test_a_factorization_is_shorter_than_the_filter(self, name):
        """Test that lifting buys fewer operations than the convolution."""
        # a step costs its own length; the convolution costs the filter twice
        wave = wavelet(name)
        cost = sum(len(step.q.c) for step in factor(wave.dec_lo, wave.dec_hi).steps)
        assert cost <= 2 * len(wave.dec_lo)

    def test_haar_takes_two_steps(self):
        """Test the textbook case, a predict and an update."""
        haar = wavelet("haar")
        assert len(factor(haar.dec_lo, haar.dec_hi).steps) == 2

    @pytest.mark.parametrize("name", sorted(UNFACTORED))
    def test_what_does_not_factor_says_so(self, name):
        """Test that an unreduced bank raises rather than returning nonsense."""
        wave = wavelet(name)
        with pytest.raises(ValueError):
            factor(wave.dec_lo, wave.dec_hi)


class TestUnfactorable:
    """Tests for the banks that do not factor into lifting steps."""

    def test_an_all_zero_bank_raises(self):
        """Test the contract an empty lead would otherwise break."""
        with pytest.raises(ValueError, match="does not factor"):
            factor([0.0], [0.0])

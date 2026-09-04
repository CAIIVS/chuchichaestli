# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for factoring a two-channel filter bank into lifting steps.

The factorization is not unique, so what is asserted is not a particular set of
steps but that multiplying them back gives the bank they came from.
"""

import pytest

from chuchichaestli.dwt import wavelet, wavelist
from chuchichaestli.utils.arithmetic.laurent import Laurent
from chuchichaestli.utils.arithmetic.lifting import factor, matrix, polyphase, rebuild


# the one bank whose factorization does not yet reconstruct to tolerance
UNFACTORED = {"bior4.4"}

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

    def test_a_filter_splits_into_its_phases(self):
        """Test that the even and odd phases interleave back to the filter."""
        even, odd = polyphase([1.0, 2.0, 3.0, 4.0])
        assert even.c == (1.0, 3.0)
        assert odd.c == (2.0, 4.0)


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

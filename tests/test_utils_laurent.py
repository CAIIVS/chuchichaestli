# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for Laurent polynomial arithmetic."""

import random

import pytest

from chuchichaestli.utils.arithmetic.laurent import Laurent, divide, reverse


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


class TestLaurent:
    """Tests for the polynomial arithmetic the factorization runs on."""

    def test_a_product_adds_the_exponents(self):
        """Test that multiplication tracks where the terms land."""
        a = Laurent.of([1.0, 2.0], -3)
        b = Laurent.of([3.0], 5)
        assert (a * b).c == (3.0, 6.0)
        assert (a * b).low == 2

    def test_reversing_twice_gives_back_the_polynomial(self):
        """Test that substituting `1/z` is its own inverse."""
        p = Laurent.of([1.0, -2.0, 4.0], -2)
        assert reverse(reverse(p)) == p

    @pytest.mark.parametrize("seed", range(40))
    def test_division_leaves_a_smaller_remainder(self, seed):
        """Test that `a == q * b + r` with the remainder shorter than `b`."""
        rng = random.Random(seed)
        a = Laurent.of([rng.uniform(-2, 2) for _ in range(rng.randint(1, 7))],
                       rng.randint(-3, 3))
        b = Laurent.of([rng.uniform(-2, 2) for _ in range(rng.randint(1, 5))],
                       rng.randint(-3, 3))
        if not b:
            pytest.skip("the divisor came out zero")
        q, r = divide(a, b)
        assert apart((q * b) + r, a) < 1e-9
        assert not r or r.span < b.span


class TestReverse:
    """Reversal swaps the two ends without leaving the canonical form."""

    def test_reversing_zero_gives_canonical_zero(self):
        """Test that the zero polynomial survives a round of reversal."""
        zero = Laurent.of([0.0])
        assert reverse(zero) == zero
        assert reverse(zero).low == 0

    @pytest.mark.parametrize("low", [-3, -1, 0, 2])
    def test_reversal_is_its_own_inverse(self, low):
        """Test that reversing twice returns what went in."""
        p = Laurent.of([1.0, -2.0, 0.5], low)
        assert reverse(reverse(p)) == p

    def test_reversal_swaps_the_ends(self):
        """Test that the coefficients come back in the other order."""
        p = Laurent.of([1.0, 2.0, 3.0], -1)
        assert reverse(p).c == (3.0, 2.0, 1.0)

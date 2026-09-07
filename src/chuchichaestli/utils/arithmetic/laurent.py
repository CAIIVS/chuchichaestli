# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Laurent polynomials, the ring a two-channel filter bank factors over.

A Laurent polynomial carries negative powers too, which is what lets a delay
and an advance sit in the same ring.
"""

from collections.abc import Sequence
from dataclasses import dataclass


__all__ = ["Laurent", "divide", "reverse"]


@dataclass(frozen=True)
class Laurent:
    """A Laurent polynomial `sum_i c[i] z**(low + i)`.

    Args:
        c: Coefficients, lowest exponent first.
        low: Exponent the first coefficient carries.
    """

    c: tuple[float, ...]
    low: int = 0

    @staticmethod
    def of(values: Sequence[float], low: int = 0, scale: float = 0.0) -> "Laurent":
        """Build a trimmed polynomial.

        Args:
            values: Coefficients, lowest exponent first.
            low: Exponent the first coefficient carries.
            scale: Magnitude the coefficients came from, where that is larger
                than the coefficients left. A remainder can be residue all the
                way down, which it cannot tell from its own terms alone.
        """
        c = list(values)
        tol = 1e-11 * max(scale, max((abs(v) for v in c), default=0.0))
        while c and abs(c[0]) <= tol:
            c.pop(0)
            low += 1
        while c and abs(c[-1]) <= tol:
            c.pop()
        return Laurent(tuple(c), low if c else 0)

    def __bool__(self) -> bool:
        """Whether the polynomial has any term."""
        return bool(self.c)

    @property
    def high(self) -> int:
        """Exponent of the last coefficient."""
        return self.low + len(self.c) - 1

    @property
    def span(self) -> int:
        """Number of terms less one, the degree once normalized."""
        return len(self.c) - 1 if self.c else -1

    def __add__(self, other: "Laurent") -> "Laurent":
        """Add two polynomials."""
        if not self:
            return other
        if not other:
            return self
        low = min(self.low, other.low)
        size = max(self.high, other.high) - low + 1
        out = [0.0] * size
        for p in (self, other):
            for i, v in enumerate(p.c):
                out[p.low + i - low] += v
        return Laurent.of(out, low)

    def __mul__(self, other: "Laurent") -> "Laurent":
        """Multiply two polynomials."""
        if not self or not other:
            return Laurent((), 0)
        out = [0.0] * (len(self.c) + len(other.c) - 1)
        for i, a in enumerate(self.c):
            for j, b in enumerate(other.c):
                out[i + j] += a * b
        return Laurent.of(out, self.low + other.low)

    def __neg__(self) -> "Laurent":
        """Negate every coefficient."""
        return Laurent(tuple(-v for v in self.c), self.low)

    def scaled(self, k: float) -> "Laurent":
        """Multiply by a scalar.

        Args:
            k: Factor to apply.
        """
        return Laurent.of([v * k for v in self.c], self.low)


def reverse(p: Laurent) -> Laurent:
    """Substitute `1/z` for `z`, which swaps the two ends.

    Args:
        p: Polynomial to reverse.
    """
    if not p.c:
        return Laurent((), 0)
    return Laurent(tuple(reversed(p.c)), -p.high)


def _growth(q: Laurent, r: Laurent) -> float:
    """How much a division inflates its operands.

    The quotient multiplies the divisor back in, so a large one is what carries
    rounding error into every later step.

    Args:
        q: Quotient.
        r: Remainder.
    """
    return max((abs(v) for v in q.c + r.c), default=0.0)


def _divide_high(a: Laurent, b: Laurent) -> tuple[Laurent, Laurent]:
    """Divide, cancelling `a` from its highest term down.

    Args:
        a: Dividend.
        b: Divisor.
    """
    rest = list(a.c)
    quotient = [0.0] * max(len(rest) - len(b.c) + 1, 0)
    for i in range(len(rest) - len(b.c), -1, -1):
        share = rest[i + len(b.c) - 1] / b.c[-1]
        quotient[i] = share
        for j, v in enumerate(b.c):
            rest[i + j] -= share * v
    scale = max((abs(v) for v in a.c), default=0.0)
    return Laurent.of(quotient, a.low - b.low), Laurent.of(rest, a.low, scale)


def divide(a: Laurent, b: Laurent) -> tuple[Laurent, Laurent]:
    """Divide `a` by `b`, leaving a remainder of smaller span.

    Either end of `b` can serve as the pivot. Both are tried and the one that
    inflates the coefficients least is kept, since dividing through a small
    coefficient is what carries rounding error into every later step.

    Args:
        a: Dividend.
        b: Divisor, which may not be zero.

    Returns:
        The quotient and the remainder.

    Raises:
        ZeroDivisionError: If `b` is zero.
    """
    if not b:
        raise ZeroDivisionError("a Laurent polynomial cannot divide by zero")
    top = _divide_high(a, b)
    q, r = _divide_high(reverse(a), reverse(b))
    bottom = (reverse(q), reverse(r))
    return min(top, bottom, key=lambda pair: _growth(*pair))

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Factor a two-channel filter bank into lifting steps.

A lifting factorization rewrites the transform as alternating in-place updates
of the even and odd samples, which needs fewer multiplications than the
convolution and no scratch buffer. Any bank whose polyphase determinant is a
monomial factors, wavelets among them.

The factorization is the Euclidean algorithm on the polyphase matrix, after
Daubechies and Sweldens, "Factoring wavelet transforms into lifting steps"
(1998). It is not unique; any factorization whose product is the original
matrix serves.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from chuchichaestli.utils.arithmetic.laurent import Laurent, divide


__all__ = ["Lifting", "Step", "factor", "matrix", "polyphase", "rebuild"]


def polyphase(filt: Sequence[float]) -> tuple[Laurent, Laurent]:
    """Split a filter into its even and odd phases.

    Args:
        filt: Filter coefficients.

    Returns:
        The even and the odd phase.
    """
    return Laurent.of(list(filt[0::2])), Laurent.of(list(filt[1::2]))


@dataclass(frozen=True)
class Step:
    """One lifting step, adding a filtered channel onto the other.

    Args:
        on_detail: Whether the detail channel is the one updated.
        q: Filter applied to the channel that is read.
    """

    on_detail: bool
    q: Laurent


@dataclass(frozen=True)
class Lifting:
    """A filter bank as a scaling followed by lifting steps.

    Args:
        steps: Steps in the order they apply to a signal.
        approx: Factor and delay the approximation channel starts from.
        detail: Factor and delay the detail channel starts from.
    """

    steps: tuple[Step, ...]
    approx: tuple[float, int]
    detail: tuple[float, int]


def matrix(
    low: Sequence[float], high: Sequence[float]
) -> tuple[Laurent, Laurent, Laurent, Laurent]:
    """The polyphase matrix of a filter bank, row by row.

    Args:
        low: Low-pass analysis filter.
        high: High-pass analysis filter.
    """
    he, ho = polyphase(low)
    ge, go = polyphase(high)
    return he, ho, ge, go


def _held(q: Laurent, r: Laurent) -> Laurent:
    """Hold one term back when the division would come out exact.

    An exact division clears the row it divides, which leaves the matrix on the
    wrong diagonal and costs three further steps to swap back. Keeping the
    lowest term of the quotient leaves a monomial there instead, which the next
    step clears from the side that ends triangular.

    Args:
        q: Quotient the division gave.
        r: Remainder it left.
    """
    if r or len(q.c) <= 1:
        return q
    return Laurent(q.c[1:], q.low + 1)


def _apart(a: Laurent, b: Laurent) -> float:
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


def _monomial(p: Laurent) -> tuple[float, int]:
    """Read a one-term polynomial as a factor and a delay.

    Args:
        p: Polynomial expected to have a single term.

    Raises:
        ValueError: If `p` is not a single term.
    """
    if len(p.c) != 1:
        raise ValueError(f"expected a single term, got {len(p.c)}")
    return p.c[0], p.low


def factor(low: Sequence[float], high: Sequence[float]) -> Lifting:
    """Factor a filter bank into lifting steps.

    Row operations on the polyphase matrix are lifting steps, so reducing it to
    a diagonal records the factorization and the inverse operations, taken in
    reverse, rebuild the bank. The row of larger span is always the one
    divided, and on a tie the detail row, which is the one driven to zero.

    Args:
        low: Low-pass analysis filter.
        high: High-pass analysis filter.

    Returns:
        The scaling and the steps that follow it.

    Raises:
        ValueError: If the bank does not reduce to a diagonal.
    """
    he, ho, ge, go = matrix(low, high)
    row1, row2 = [he, ho], [ge, go]
    one = Laurent((1.0,), 0)
    undone: list[Step] = []

    def take(on_detail: bool, q: Laurent) -> None:
        """Subtract `q` times one row from the other, and remember it."""
        nonlocal row1, row2
        if on_detail:
            row2 = [row2[0] + (-q * row1[0]), row2[1] + (-q * row1[1])]
        else:
            row1 = [row1[0] + (-q * row2[0]), row1[1] + (-q * row2[1])]
        undone.append(Step(on_detail, q))

    for _ in range(8 * (len(low) + 2)):
        if not row2[0] or not row1[0]:
            break
        if row1[0].span > row2[0].span:
            take(False, _held(*divide(row1[0], row2[0])) or one)
        else:
            take(True, _held(*divide(row2[0], row1[0])) or one)
    if row2[0] and row1[0]:
        raise ValueError("the bank did not reduce to a triangular form")
    if not row1[0]:
        # the reduction cleared the other corner, leaving `[[0, b], [c, d]]`;
        # `b` and `c` are monomials there, so three more steps diagonalize it
        if row2[1]:
            q, _ = divide(row2[1], row1[1])
            take(True, q)
        lead = row2[0]
        take(False, -Laurent((1.0 / lead.c[0],), -lead.low))
        take(True, lead)
    if row1[1]:
        q, _ = divide(row1[1], row2[1])
        take(False, q)
    lifting = Lifting(
        tuple(reversed(undone)), _monomial(row1[0]), _monomial(row2[1])
    )
    # the reduction can end on a matrix that is only nearly diagonal, and a
    # factorization that does not multiply back is worse than none
    if max(_apart(g, w) for g, w in zip(rebuild(lifting), (he, ho, ge, go))) > 1e-9:
        raise ValueError("the bank did not factor into lifting steps")
    return lifting


def rebuild(lifting: Lifting) -> tuple[Laurent, Laurent, Laurent, Laurent]:
    """Multiply a factorization back into a polyphase matrix.

    Args:
        lifting: Factorization to expand.

    Returns:
        The matrix, row by row.
    """
    ka, sa = lifting.approx
    kd, sd = lifting.detail
    zero = Laurent((), 0)
    row1 = [Laurent((ka,), sa), zero]
    row2 = [zero, Laurent((kd,), sd)]
    for step in lifting.steps:
        if step.on_detail:
            row2 = [row2[0] + step.q * row1[0], row2[1] + step.q * row1[1]]
        else:
            row1 = [row1[0] + step.q * row2[0], row1[1] + step.q * row2[1]]
    return row1[0], row1[1], row2[0], row2[1]

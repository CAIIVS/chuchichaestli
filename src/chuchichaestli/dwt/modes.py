# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Signal extension modes for the boundary handling of the wavelet transform."""

import torch
from torch.nn import functional as F

from chuchichaestli.utils import view_along_axis
from functools import lru_cache
from typing import Literal


__all__ = [
    "ExtensionModeTypes",
    "MODE_TO_CODE",
    "extension_indices",
    "pad_signal",
    "pad_signal_adjoint",
]


ExtensionModeTypes = Literal[
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "periodization",
    "antisymmetric",
    "antireflect",
]

# Mirrored by the `PadMode` enum in `csrc/common/boundary.h`
MODE_TO_CODE: dict[str, int] = {
    "zero": 0,
    "constant": 1,
    "symmetric": 2,
    "reflect": 3,
    "periodic": 4,
    "periodization": 5,
    "antisymmetric": 6,
    "antireflect": 7,
}


def _sample(i: int, n: int, mode: str) -> tuple[float, int, float, float]:
    """Express an out-of-range sample through the samples that exist.

    Returns the quadruple `(sign, index, lo, hi)` standing for the value
    `sign * x[index] + lo * x[0] + hi * x[n - 1]`.

    Args:
        i: Index to resolve; may lie outside `[0, n)`.
        n: Length of the signal.
        mode: Signal extension mode.

    Raises:
        ValueError: If `mode` is not a known extension mode.
    """
    if 0 <= i < n:
        return (1.0, i, 0.0, 0.0)
    if n == 1 and mode in ("reflect", "antireflect"):
        # whole-sample folding needs two samples to step between
        return (1.0, 0, 0.0, 0.0)
    match mode:
        case "zero":
            return (0.0, 0, 0.0, 0.0)
        case "constant":
            return (1.0, 0 if i < 0 else n - 1, 0.0, 0.0)
        case "periodic" | "periodization":
            return _sample(i % n, n, mode)
        case "symmetric":
            return _sample(-i - 1 if i < 0 else 2 * n - 1 - i, n, mode)
        case "reflect":
            return _sample(-i if i < 0 else 2 * (n - 1) - i, n, mode)
        case "antisymmetric":
            sign, idx, lo, hi = _sample(-i - 1 if i < 0 else 2 * n - 1 - i, n, mode)
            return (-sign, idx, -lo, -hi)
        case "antireflect":
            sign, idx, lo, hi = _sample(-i if i < 0 else 2 * (n - 1) - i, n, mode)
            anchor = (2.0, 0.0) if i < 0 else (0.0, 2.0)
            return (-sign, idx, anchor[0] - lo, anchor[1] - hi)
        case _:
            raise ValueError(
                f"Unsupported signal extension mode: {mode!r}."
                f" Use one of {sorted(MODE_TO_CODE)}."
            )


@lru_cache(maxsize=512)
def extension_indices(
    n: int, pad_lo: int, pad_hi: int, mode: ExtensionModeTypes
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the gather that extends a signal of length `n` past both ends.

    The result is cached on CPU; callers move it to the device they need.

    Args:
        n: Length of the signal.
        pad_lo: Number of samples prepended.
        pad_hi: Number of samples appended.
        mode: Signal extension mode.

    Raises:
        ValueError: If `n` is not positive or a padding is negative.
    """
    if n < 1:
        raise ValueError(f"A signal needs at least one sample; got {n}.")
    if pad_lo < 0 or pad_hi < 0:
        raise ValueError(
            f"Signal extension widths must not be negative; got ({pad_lo}, {pad_hi})."
        )
    entries = [_sample(i, n, mode) for i in range(-pad_lo, n + pad_hi)]
    sign, index, lo, hi = zip(*entries)
    return (
        torch.tensor(sign, dtype=torch.float64),
        torch.tensor(index, dtype=torch.int64),
        torch.tensor(lo, dtype=torch.float64),
        torch.tensor(hi, dtype=torch.float64),
    )


def pad_signal(
    x: torch.Tensor,
    axis: int,
    pad_lo: int,
    pad_hi: int,
    mode: ExtensionModeTypes = "zero",
) -> torch.Tensor:
    """Extend a tensor along one axis by the given signal extension mode.

    Args:
        x: Input tensor.
        axis: Axis to extend, given as a non-negative or negative index.
        pad_lo: Number of samples prepended.
        pad_hi: Number of samples appended.
        mode: Signal extension mode.
    """
    if pad_lo == 0 and pad_hi == 0:
        return x
    axis = axis % x.ndim
    if mode == "zero":
        # `F.pad` reads its widths from the last axis backwards.
        widths = [0, 0] * (x.ndim - 1 - axis) + [pad_lo, pad_hi]
        return F.pad(x, widths, mode="constant", value=0.0)

    n = x.shape[axis]
    sign, index, lo, hi = extension_indices(n, pad_lo, pad_hi, mode)
    index = index.to(x.device)
    out = x.index_select(axis, index)
    if not bool(torch.all(sign == 1.0)):
        out = out * view_along_axis(sign.to(x.dtype), x.ndim, axis).to(x.device)
    if bool(torch.any(lo != 0.0)) or bool(torch.any(hi != 0.0)):
        edge_lo = x.narrow(axis, 0, 1)
        edge_hi = x.narrow(axis, n - 1, 1)
        out = out + edge_lo * view_along_axis(lo.to(x.dtype), x.ndim, axis).to(x.device)
        out = out + edge_hi * view_along_axis(hi.to(x.dtype), x.ndim, axis).to(x.device)
    return out


def pad_signal_adjoint(
    grad: torch.Tensor,
    axis: int,
    pad_lo: int,
    pad_hi: int,
    mode: ExtensionModeTypes = "zero",
    length: int | None = None,
) -> torch.Tensor:
    """Transpose of `pad_signal`, scattering a gradient back onto the samples.

    The extension gathers, so its transpose scatters: every extended sample
    returns its share to the source it was read from, and to the two edges the
    anchored modes lean on.

    Args:
        grad: Gradient with respect to the extended tensor.
        axis: Axis that was extended.
        pad_lo: Number of samples that were prepended.
        pad_hi: Number of samples that were appended.
        mode: Signal extension mode that was used.
        length: Length of the axis before extension; derived if omitted.
    """
    axis = axis % grad.ndim
    if pad_lo == 0 and pad_hi == 0:
        return grad
    n = grad.shape[axis] - pad_lo - pad_hi if length is None else length
    if mode == "zero":
        return grad.narrow(axis, pad_lo, n)

    sign, index, lo, hi = extension_indices(n, pad_lo, pad_hi, mode)
    index = index.to(grad.device)
    contribution = grad * view_along_axis(sign.to(grad.dtype), grad.ndim, axis).to(
        grad.device
    )
    shape = list(grad.shape)
    shape[axis] = n
    out = torch.zeros(shape, dtype=grad.dtype, device=grad.device)
    out = out.index_add(axis, index, contribution)
    if bool(torch.any(lo != 0.0)) or bool(torch.any(hi != 0.0)):
        weights_lo = view_along_axis(lo.to(grad.dtype), grad.ndim, axis).to(grad.device)
        weights_hi = view_along_axis(hi.to(grad.dtype), grad.ndim, axis).to(grad.device)
        edge_lo = (grad * weights_lo).sum(dim=axis, keepdim=True)
        edge_hi = (grad * weights_hi).sum(dim=axis, keepdim=True)
        # a single-sample axis makes both edges the same one, which `index_add`
        # handles by accumulating the two contributions
        edges = torch.tensor([0, n - 1], device=grad.device)
        out = out.index_add(axis, edges, torch.cat((edge_lo, edge_hi), dim=axis))
    return out

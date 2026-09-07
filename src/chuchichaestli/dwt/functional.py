# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Separable N-dimensional discrete wavelet transforms.

Subbands carry one character per transformed axis, `'a'` for the approximation
and `'d'` for the detail branch, so a two-dimensional transform yields `aa`,
`ad`, `da` and `dd` (the classical `LL`, `LH`, `HL`, `HH`).
"""

import threading

import torch

from chuchichaestli.dwt.modes import ExtensionModeTypes, pad_signal
from chuchichaestli.dwt.wavelet import Wavelet, wavelet as as_wavelet
from chuchichaestli.models.maps import DIM_TO_CONV_FN_MAP, DIM_TO_CONVT_FN_MAP
from chuchichaestli.utils import as_inexact
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from math import prod


__all__ = [
    "dwt",
    "dwt_coeff_len",
    "dwt_max_level",
    "dwtn",
    "dwtn_approx",
    "idwt",
    "idwtn",
    "subband_keys",
    "wavedec",
    "wavedecn",
    "waverec",
    "waverecn",
]


@lru_cache(maxsize=64)
def subband_keys(dimensions: int) -> tuple[str, ...]:
    """Subband names of a `dimensions`-dimensional transform, in output order.

    Args:
        dimensions: Number of transformed axes.
    """
    keys = ("",)
    for _ in range(dimensions):
        keys = tuple(k + branch for k in keys for branch in "ad")
    return keys


def dwt_max_level(data_len: int, filter_len: int) -> int:
    """Number of levels a signal supports before it is shorter than the filter.

    Args:
        data_len: Length of the signal.
        filter_len: Length of the decomposition filters.
    """
    if filter_len < 2 or data_len < filter_len - 1:
        return 0
    return int(torch.log2(torch.tensor(data_len / (filter_len - 1.0))).item())


def dwt_coeff_len(
    data_len: int, filter_len: int, mode: ExtensionModeTypes = "zero"
) -> int:
    """Number of coefficients one level of the transform produces per subband.

    Args:
        data_len: Length of the signal.
        filter_len: Length of the decomposition filters.
        mode: Signal extension mode.

    Raises:
        ValueError: If `data_len` or `filter_len` is not positive.
    """
    if data_len < 1 or filter_len < 1:
        raise ValueError(
            f"Signal and filter must be non-empty; got {data_len} and {filter_len}."
        )
    if mode == "periodization":
        return (data_len + 1) // 2
    return (data_len + filter_len - 1) // 2


def _resolve_axes(ndim: int, axes: Sequence[int] | None) -> tuple[int, ...]:
    """Normalize transform axes to distinct non-negative indices.

    Args:
        ndim: Rank of the tensor being transformed.
        axes: Axes to transform; the trailing one if omitted.

    Raises:
        ValueError: If an axis is out of range, repeated, or if more than three
            axes are requested (`torch` has no convolution beyond three dims).
    """
    if axes is None:
        axes = (-1,)
    resolved = []
    for axis in axes:
        if not -ndim <= axis < ndim:
            raise ValueError(
                f"Axis {axis} is out of range for a {ndim}-dimensional tensor."
            )
        resolved.append(axis % ndim)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Transform axes must be distinct; got {tuple(axes)}.")
    if not 1 <= len(resolved) <= 3:
        raise ValueError(
            f"The transform runs over one to three axes; got {len(resolved)}."
        )
    return tuple(resolved)


def _fold(x: torch.Tensor, axes: Sequence[int]) -> tuple[torch.Tensor, tuple, list]:
    """Move the transform axes last and fold everything else into a batch axis.

    Args:
        x: Input tensor.
        axes: Transform axes, already normalized.

    Returns:
        The folded tensor, the leading shape, and the permutation applied.
    """
    others = [i for i in range(x.ndim) if i not in axes]
    perm = others + list(axes)
    moved = x.permute(perm) if perm != list(range(x.ndim)) else x
    lead = tuple(moved.shape[: len(others)])
    spatial = moved.shape[len(others) :]
    return moved.reshape(-1, 1, *spatial), lead, perm


def _unfold(band: torch.Tensor, lead: tuple, perm: list) -> torch.Tensor:
    """Undo `_fold` for a single subband.

    Args:
        band: One subband, shaped `(B, *spatial)`.
        lead: Leading shape `_fold` folded away.
        perm: Permutation `_fold` applied.
    """
    out = band.reshape(*lead, *band.shape[1:])
    inverse = [perm.index(i) for i in range(len(perm))]
    return out.permute(inverse) if perm != sorted(perm) else out


def _bank(
    lo: torch.Tensor, hi: torch.Tensor, groups: int, axis: int, dimensions: int
) -> torch.Tensor:
    """Build a kernel that is degenerate on every axis but `axis`.

    Args:
        lo: Low-pass filter.
        hi: High-pass filter.
        groups: Number of input channels, each transformed independently.
        axis: Spatial axis the filter runs along.
        dimensions: Number of spatial axes.
    """
    shape = [1] * dimensions
    shape[axis] = lo.numel()
    pair = torch.stack((lo, hi)).reshape(2, 1, *shape)
    return pair.repeat(groups, 1, *([1] * dimensions))


def _pad_for_decomposition(
    h: torch.Tensor, axis: int, filter_len: int, mode: ExtensionModeTypes
) -> torch.Tensor:
    """Extend one spatial axis by what the decomposition convolution consumes.

    Args:
        h: Input tensor, shaped `(B, G, *spatial)`.
        axis: Spatial axis to extend.
        filter_len: Length of the decomposition filters.
        mode: Signal extension mode.
    """
    length = h.shape[2 + axis]
    if mode == "periodization":
        if length % 2:
            # An odd axis is made even by repeating its last sample.
            h = pad_signal(h, 2 + axis, 0, 1, "constant")
        return pad_signal(h, 2 + axis, filter_len // 2 - 1, filter_len // 2, "periodic")
    return pad_signal(h, 2 + axis, filter_len - 2, filter_len - 2 + (length % 2), mode)


def _decompose(
    h: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    axis: int,
    mode: ExtensionModeTypes,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split every band of `h` along one spatial axis into a low- and high-pass half.

    Args:
        h: Bands so far, shaped `(B, G, *spatial)`.
        lo: Decomposition low-pass filter.
        hi: Decomposition high-pass filter.
        axis: Spatial axis to transform.
        mode: Signal extension mode.
        out: Storage the kernel writes into, when the caller keeps one across
            axes; ignored by the pure-torch path.
    """
    from chuchichaestli.dwt import _ext

    dimensions = h.ndim - 2
    groups = h.shape[1]
    filter_len = lo.numel()
    length = h.shape[2 + axis]
    if mode == "periodization" and length % 2:
        # an odd axis is made even by repeating its last sample
        h = pad_signal(h, 2 + axis, 0, 1, "constant")
        length += 1
    if mode == "periodization":
        pad_lo, pad_hi = filter_len // 2 - 1, filter_len // 2
    else:
        pad_lo, pad_hi = filter_len - 2, filter_len - 2 + (length % 2)
    out_length = (length + pad_lo + pad_hi - filter_len) // 2 + 1

    if _ext.kernels_available(h.device, h.dtype):
        return _ext.dwt_axis(h, lo, hi, axis, mode, pad_lo, pad_hi, out_length, out)

    extension = "periodic" if mode == "periodization" else mode
    h = pad_signal(h, 2 + axis, pad_lo, pad_hi, extension)
    # `conv` cross-correlates, so the filters are flipped to convolve.
    weight = _bank(lo.flip(0), hi.flip(0), groups, axis, dimensions)
    stride = [1] * dimensions
    stride[axis] = 2
    return DIM_TO_CONV_FN_MAP[dimensions](h, weight, stride=stride, groups=groups)


def _decompose_lowpass(
    h: torch.Tensor, lo: torch.Tensor, axis: int, mode: ExtensionModeTypes
) -> torch.Tensor:
    """Keep only the low-pass half of every band of `h` along one spatial axis.

    Args:
        h: Bands so far, shaped `(B, G, *spatial)`.
        lo: Decomposition low-pass filter.
        axis: Spatial axis to transform.
        mode: Signal extension mode.
    """
    from chuchichaestli.dwt import _ext

    dimensions = h.ndim - 2
    groups = h.shape[1]
    filter_len = lo.numel()
    if _ext.lowpass_kernel_applies(h.device, h.dtype):
        length = h.shape[2 + axis]
        if mode == "periodization" and length % 2:
            h = pad_signal(h, 2 + axis, 0, 1, "constant")
            length += 1
        if mode == "periodization":
            pad_lo, pad_hi = filter_len // 2 - 1, filter_len // 2
        else:
            pad_lo, pad_hi = filter_len - 2, filter_len - 2 + (length % 2)
        out_length = (length + pad_lo + pad_hi - filter_len) // 2 + 1
        return _ext.dwt_lowpass_axis(
            h, lo, axis, mode, pad_lo, pad_hi, out_length
        )
    h = _pad_for_decomposition(h, axis, filter_len, mode)
    shape = [1] * dimensions
    shape[axis] = filter_len
    weight = lo.flip(0).reshape(1, 1, *shape).repeat(groups, 1, *([1] * dimensions))
    stride = [1] * dimensions
    stride[axis] = 2
    return DIM_TO_CONV_FN_MAP[dimensions](h, weight, stride=stride, groups=groups)


def _reconstruct(
    h: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    axis: int,
    mode: ExtensionModeTypes,
    length: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge low- and high-pass band pairs of `h` back along one spatial axis.

    Args:
        h: Band pairs, shaped `(B, 2 G, *spatial)`.
        lo: Reconstruction low-pass filter.
        hi: Reconstruction high-pass filter.
        axis: Spatial axis to reconstruct.
        mode: Signal extension mode the decomposition used.
        length: Length the reconstructed axis is trimmed to.
        out: Storage the kernel writes into, when the caller keeps one across
            axes; ignored by the pure-torch path.
    """
    dimensions = h.ndim - 2
    groups = h.shape[1] // 2
    filter_len = lo.numel()
    from chuchichaestli.dwt import _ext

    if _ext.idwt_kernel_applies(h.device) and _ext.kernels_available(
        h.device, h.dtype
    ):
        trim = filter_len // 2 - 1 if mode == "periodization" else filter_len - 2
        return _ext.idwt_axis(h, lo, hi, axis, mode, trim, length, out)
    weight = _bank(lo, hi, groups, axis, dimensions)
    stride = [1] * dimensions
    stride[axis] = 2
    out = DIM_TO_CONVT_FN_MAP[dimensions](h, weight, stride=stride, groups=groups)
    if mode == "periodization":
        # Critically sampled: what spills past either end belongs to the other.
        even = 2 * h.shape[2 + axis]
        out = _wrap(out, 2 + axis, filter_len // 2 - 1, even)
        return out.narrow(2 + axis, 0, length)
    return out.narrow(2 + axis, filter_len - 2, length)


def _wrap(x: torch.Tensor, axis: int, trim: int, length: int) -> torch.Tensor:
    """Add the overhang of a circular reconstruction back in before trimming.

    Args:
        x: Reconstruction including its overhang.
        axis: Axis to fold and trim.
        trim: Number of samples the reconstruction leads by.
        length: Length the axis is trimmed to.
    """
    total = x.shape[axis]
    index = (torch.arange(total, device=x.device) - trim) % length
    shape = list(x.shape)
    shape[axis] = length
    zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return zeros.index_add(axis, index, x)


def _HAAR_ADJOINT(grad: torch.Tensor, dimensions: int) -> torch.Tensor:
    """Adjoint of the fused Haar decomposition, used by its backward pass.

    The transform is orthogonal, so the adjoint is the reconstruction, which
    the fused kernel already carries.

    Args:
        grad: Gradient with respect to the stacked subbands.
        dimensions: Number of spatial axes.
    """
    from chuchichaestli.dwt import _ext

    haar = as_wavelet("haar")
    _, _, rec_lo, rec_hi = haar.filters(grad.dtype, grad.device)
    lengths = tuple(2 * grad.shape[2 + axis] for axis in range(dimensions))
    if _ext.idwt_nd_applies(grad.device, dimensions) and _ext.kernels_available(
        grad.device, grad.dtype
    ):
        return _ext.idwt_nd(
            grad, rec_lo, rec_hi, "zero", (0,) * dimensions, lengths
        )
    h = grad
    for axis in reversed(range(dimensions)):
        h = _reconstruct(h, rec_lo, rec_hi, axis, "zero", 2 * h.shape[2 + axis])
    return h


def dwtn(
    data: torch.Tensor,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axes: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Single-level separable wavelet transform over one or more axes.

    Args:
        data: Input tensor of any rank.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        axes: Axes to transform; the trailing one if omitted.

    Returns:
        The subbands, keyed by their `'a'`/`'d'` names.
    """
    data = as_inexact(data)
    axes = _resolve_axes(data.ndim, axes)
    wavelet = as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = wavelet.filters(data.dtype, data.device)
    from chuchichaestli.dwt import _ext

    h, lead, perm = _fold(data, axes)
    spatial = tuple(h.shape[2:])
    if _ext.kernels_available(h.device, h.dtype) and _ext.fused_haar_applies(
        wavelet, mode, spatial
    ):
        h = _ext.haar_nd(h, len(axes))
    else:
        h = _decompose_axes(h, dec_lo, dec_hi, len(axes), mode)
    keys = subband_keys(len(axes))
    return {key: _unfold(h[:, i], lead, perm) for i, key in enumerate(keys)}


_SCRATCH = threading.local()
_SCRATCH_LIMIT = 8
_STACK_SLOT = -1


def _scratch(
    slot: int, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Lend storage of at least this size back on every call that asks for it.

    One buffer per slot rather than one per shape, grown to the largest a
    caller has asked that slot for and handed back as a view. A caller working
    through many shapes keeps reusing it, where a buffer per shape would hold
    only the last few and drop them just as they came round again.

    Held per thread, so two transforms running at once never share one.

    Args:
        slot: Which of a transform's intermediates this is; a spatial axis
            index for the ones a decomposition hands between axes, and
            `_STACK_SLOT` for the subband stack a reconstruction reads.
        shape: Shape the intermediate takes.
        dtype: Type the intermediate takes.
        device: Device the intermediate lives on.
    """
    pool = getattr(_SCRATCH, "pool", None)
    if pool is None:
        pool = _SCRATCH.pool = OrderedDict()
    key = (slot, dtype, device)
    needed = prod(shape)
    buffer = pool.get(key)
    if buffer is None or buffer.numel() < needed:
        if buffer is None and len(pool) >= _SCRATCH_LIMIT:
            pool.popitem(last=False)
        buffer = torch.empty(needed, dtype=dtype, device=device)
        pool[key] = buffer
        pool.move_to_end(key)
    if buffer.numel() == needed:
        return buffer.view(shape)
    return buffer[:needed].view(shape)


def _decompose_axes(
    h: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    dimensions: int,
    mode: ExtensionModeTypes,
) -> torch.Tensor:
    """Split every band along every spatial axis in turn.

    Each axis writes a tensor the next one reads and then drops. Left to the
    allocator that returns freshly mapped pages every call, and faulting them
    in can cost more than the transform; two buffers handed back and forth
    are mapped once.

    Args:
        h: Bands so far, shaped `(batch, groups, spatial...)`.
        lo: Decomposition low-pass filter.
        hi: Decomposition high-pass filter.
        dimensions: Number of spatial axes to transform.
        mode: Signal extension mode.
    """
    from chuchichaestli.dwt import _ext

    # only the host allocator maps a fresh page for every intermediate; the
    # accelerator hands its own storage back and takes no buffer from here
    if (
        dimensions < 2
        or h.device.type != "cpu"
        or not _ext.kernels_available(h.device, h.dtype)
    ):
        for axis in range(dimensions):
            h = _decompose(h, lo, hi, axis, mode)
        return h

    filter_len = lo.numel()
    shape = list(h.shape)
    for axis in range(dimensions):
        length = shape[2 + axis]
        if mode == "periodization":
            length += length % 2
            out_length = length // 2
        else:
            pad_lo = filter_len - 2
            pad_hi = pad_lo + (length % 2)
            out_length = (length + pad_lo + pad_hi - filter_len) // 2 + 1
        shape[1] *= 2
        shape[2 + axis] = out_length
        # the last axis writes what the caller keeps, so only the ones before
        # it are handed a buffer to reuse
        out = (
            None
            if axis == dimensions - 1
            else _scratch(axis, tuple(shape), h.dtype, h.device)
        )
        h = _decompose(h, lo, hi, axis, mode, out)
    return h


def dwtn_approx(
    data: torch.Tensor,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axes: Sequence[int] | None = None,
) -> torch.Tensor:
    """Approximation band of a single-level transform.

    The compiled kernel splits every band at once for less than a convolution
    costs to pad for, so it is taken where it serves and its detail bands
    dropped; the convolution only skips them where there is no kernel.

    Args:
        data: Input tensor of any rank.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        axes: Axes to transform; the trailing one if omitted.
    """
    data = as_inexact(data)
    axes = _resolve_axes(data.ndim, axes)
    dec_lo, _, _, _ = as_wavelet(wavelet).filters(data.dtype, data.device)
    h, lead, perm = _fold(data, axes)
    for axis in range(len(axes)):
        h = _decompose_lowpass(h, dec_lo, axis, mode)
    return _unfold(h[:, 0], lead, perm)


def _stack_bands(bands: list[torch.Tensor]) -> torch.Tensor:
    """Lay every subband into one tensor for the reconstruction to read.

    Args:
        bands: The folded subbands, in subband-key order.
    """
    first = bands[0]
    if first.device.type != "cpu" or (
        torch.is_grad_enabled() and any(band.requires_grad for band in bands)
    ):
        return torch.cat(bands, dim=1)
    shape = (
        first.shape[0],
        sum(band.shape[1] for band in bands),
        *first.shape[2:],
    )
    out = _scratch(_STACK_SLOT, shape, first.dtype, first.device)
    return torch.cat(bands, dim=1, out=out)


def idwtn(
    coeffs: dict[str, torch.Tensor],
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axes: Sequence[int] | None = None,
    output_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Invert a single-level separable wavelet transform.

    Args:
        coeffs: Subbands keyed by their `'a'`/`'d'` names.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode the decomposition used.
        axes: Axes that were transformed; the trailing one if omitted.
        output_size: Length of each transformed axis in the reconstruction;
            an even length is assumed if omitted.

    Raises:
        ValueError: If a subband is missing or the subbands disagree in shape.
    """
    sample = next(iter(coeffs.values()))
    axes = _resolve_axes(sample.ndim, axes)
    keys = subband_keys(len(axes))
    missing = [key for key in keys if key not in coeffs]
    if missing:
        raise ValueError(
            f"Missing subband(s) {missing} for a {len(axes)}-dimensional transform;"
            f" expected {list(keys)}."
        )
    shapes = {tuple(coeffs[key].shape) for key in keys}
    if len(shapes) != 1:
        raise ValueError(f"All subbands must have the same shape; got {sorted(shapes)}.")

    wavelet = as_wavelet(wavelet)
    stacked = [_fold(as_inexact(coeffs[key]), axes) for key in keys]
    lead, perm = stacked[0][1], stacked[0][2]
    bands = [band for band, _, _ in stacked]
    h = _stack_bands(bands)
    _, _, rec_lo, rec_hi = wavelet.filters(h.dtype, h.device)

    sizes = _reconstruction_sizes(h.shape[2:], wavelet.filter_len, mode, output_size)
    from chuchichaestli.dwt import _ext

    if _ext.idwt_nd_applies(h.device, len(axes)) and _ext.kernels_available(
        h.device, h.dtype
    ):
        filter_len = wavelet.filter_len
        trim = filter_len // 2 - 1 if mode == "periodization" else filter_len - 2
        fused = _ext.idwt_nd(
            h, rec_lo, rec_hi, mode, (trim,) * len(axes), tuple(sizes)
        )
        return _unfold(fused[:, 0], lead, perm)
    # Decomposition appends one character per axis, so the axis transformed last
    # varies fastest and its band pairs are adjacent.
    for axis in reversed(range(len(axes))):
        h = _reconstruct(h, rec_lo, rec_hi, axis, mode, sizes[axis])
    return _unfold(h[:, 0], lead, perm)


def _reconstruction_sizes(
    coeff_shape: Sequence[int],
    filter_len: int,
    mode: ExtensionModeTypes,
    output_size: Sequence[int] | None,
) -> list[int]:
    """Length each transformed axis is reconstructed to.

    Args:
        coeff_shape: Spatial shape of one subband.
        filter_len: Length of the filters.
        mode: Signal extension mode the decomposition used.
        output_size: Explicit lengths, if the caller recorded them.

    Raises:
        ValueError: If `output_size` does not cover every transformed axis.
    """
    if output_size is not None:
        if len(output_size) != len(coeff_shape):
            raise ValueError(
                f"`output_size` must give one length per transformed axis;"
                f" got {len(output_size)} for {len(coeff_shape)} axes."
            )
        return [int(s) for s in output_size]
    if mode == "periodization":
        return [2 * int(n) for n in coeff_shape]
    return [2 * int(n) - filter_len + 2 for n in coeff_shape]


def dwt(
    data: torch.Tensor,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-level wavelet transform along one axis.

    Args:
        data: Input tensor of any rank.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        axis: Axis to transform.

    Returns:
        The approximation and the detail coefficients.
    """
    bands = dwtn(data, wavelet, mode, (axis,))
    return bands["a"], bands["d"]


def idwt(
    approx: torch.Tensor,
    detail: torch.Tensor,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axis: int = -1,
    output_size: int | None = None,
) -> torch.Tensor:
    """Invert a single-level wavelet transform along one axis.

    Args:
        approx: Approximation coefficients.
        detail: Detail coefficients.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode the decomposition used.
        axis: Axis that was transformed.
        output_size: Length of the reconstructed axis.
    """
    sizes = None if output_size is None else (output_size,)
    return idwtn({"a": approx, "d": detail}, wavelet, mode, (axis,), sizes)


def wavedecn(
    data: torch.Tensor,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    level: int | None = None,
    axes: Sequence[int] | None = None,
) -> list:
    """Multi-level separable wavelet transform, recursing on the approximation.

    Args:
        data: Input tensor of any rank.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        level: Number of levels; the maximum the shortest axis supports if omitted.
        axes: Axes to transform; the trailing one if omitted.

    Returns:
        `[approx, details_level, ..., details_1]`, coarsest first, each `details`
        holding every subband but the all-approximation one.

    Raises:
        ValueError: If `level` is negative.
    """
    data = as_inexact(data)
    axes = _resolve_axes(data.ndim, axes)
    wavelet = as_wavelet(wavelet)
    if level is None:
        level = min(
            dwt_max_level(data.shape[axis], wavelet.dec_len) for axis in axes
        )
        level = max(level, 1)
    if level < 0:
        raise ValueError(f"The number of levels must not be negative; got {level}.")

    from chuchichaestli.dwt import _ext

    approx_key = "a" * len(axes)
    keys = subband_keys(len(axes))
    result: list = []

    folded, lead, perm = _fold(data, axes)
    shapes = [tuple(folded.shape[2:])]
    for _ in range(level - 1):
        shapes.append(
            tuple(dwt_coeff_len(n, wavelet.dec_len, mode) for n in shapes[-1])
        )
    compiled = level >= 1 and not (
        torch.is_grad_enabled() and data.requires_grad
    ) and _ext.kernels_available(folded.device, folded.dtype)

    # the kernel runs the recursion: the Haar form where every level suits it,
    # and the general one otherwise
    levels = None
    if compiled:
        if all(
            _ext.fused_haar_applies(wavelet, mode, shape) for shape in shapes
        ):
            levels = _ext.haar_wavedec(folded, len(axes), level)
        elif _ext.fused_recursion_applies(mode, shapes):
            dec_lo, dec_hi, _, _ = wavelet.filters(folded.dtype, folded.device)
            levels = _ext.wavedec_axes(folded, dec_lo, dec_hi, mode, level)
    if levels is not None:
        for stacked in levels:
            bands = {
                key: _unfold(stacked[:, i], lead, perm)
                for i, key in enumerate(keys)
            }
            approx = bands.pop(approx_key)
            result.append(bands)
        result.append(approx)
        return result[::-1]

    approx = data
    for _ in range(level):
        bands = dwtn(approx, wavelet, mode, axes)
        approx = bands.pop(approx_key)
        result.append(bands)
    result.append(approx)
    return result[::-1]


def _trim_to_band(
    approx: torch.Tensor, band: torch.Tensor, axes: Sequence[int]
) -> torch.Tensor:
    """Drop the odd sample a coarser level carries past its detail bands.

    An axis of odd length decomposes to bands the reconstruction cannot tell
    from an even one, so a level rebuilt without a recorded size comes back one
    sample too long. That sample belongs to no detail band.

    Args:
        approx: Approximation reconstructed from the coarser level.
        band: A detail band of the level being reconstructed.
        axes: Axes that were transformed.
    """
    for axis in axes:
        if approx.shape[axis] == band.shape[axis] + 1:
            approx = approx.narrow(axis, 0, band.shape[axis])
    return approx


def waverecn(
    coeffs: list,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axes: Sequence[int] | None = None,
    output_size: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Invert a multi-level separable wavelet transform.

    Args:
        coeffs: `[approx, details_level, ..., details_1]`, as `wavedecn` returns.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode the decomposition used.
        axes: Axes that were transformed; the trailing one if omitted.
        output_size: Per level, the length of each transformed axis, coarsest
            level first.

    Raises:
        ValueError: If `coeffs` is empty or malformed.
    """
    if not coeffs:
        raise ValueError("`coeffs` must hold at least the approximation band.")
    approx, details = coeffs[0], coeffs[1:]
    if not details:
        return approx
    axes = _resolve_axes(approx.ndim, axes)
    approx_key = "a" * len(axes)
    for i, band in enumerate(details):
        sizes = None if output_size is None else output_size[i]
        if sizes is None and band:
            approx = _trim_to_band(approx, next(iter(band.values())), axes)
        approx = idwtn({approx_key: approx, **band}, wavelet, mode, axes, sizes)
    return approx


def wavedec(
    data: torch.Tensor,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    level: int | None = None,
    axis: int = -1,
) -> list[torch.Tensor]:
    """Multi-level wavelet transform along one axis.

    Args:
        data: Input tensor of any rank.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        level: Number of levels; the maximum the axis supports if omitted.
        axis: Axis to transform.

    Returns:
        `[approx, detail_level, ..., detail_1]`, coarsest first.
    """
    parts = wavedecn(data, wavelet, mode, level, (axis,))
    return [parts[0]] + [band["d"] for band in parts[1:]]


def waverec(
    coeffs: Sequence[torch.Tensor],
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    axis: int = -1,
    output_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Invert a multi-level wavelet transform along one axis.

    Args:
        coeffs: `[approx, detail_level, ..., detail_1]`, as `wavedec` returns.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode the decomposition used.
        axis: Axis that was transformed.
        output_size: Length of the axis at each level, coarsest level first.

    Raises:
        ValueError: If `coeffs` is empty.
    """
    if not coeffs:
        raise ValueError("`coeffs` must hold at least the approximation band.")
    sizes = None if output_size is None else [(n,) for n in output_size]
    parts = [coeffs[0]] + [{"d": band} for band in coeffs[1:]]
    return waverecn(parts, wavelet, mode, (axis,), sizes)

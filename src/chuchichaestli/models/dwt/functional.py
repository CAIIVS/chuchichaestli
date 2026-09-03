# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batched wavelet transforms that carry the subbands in the channel axis."""

import torch

from chuchichaestli.dwt.functional import dwtn, dwtn_approx, idwtn, subband_keys
from chuchichaestli.dwt.modes import ExtensionModeTypes
from chuchichaestli.dwt.wavelet import Wavelet
from collections.abc import Sequence
from typing import Literal


__all__ = [
    "SubbandOrderTypes",
    "dwt_nd",
    "dwt_nd_approx",
    "idwt_nd",
    "subband_names",
    "wavedec_nd",
    "waverec_nd",
]


SubbandOrderTypes = Literal["subband", "channel"]


def subband_names(dimensions: int) -> tuple[str, ...]:
    """Subband names of a `dimensions`-dimensional transform, in output order.

    Args:
        dimensions: Number of transformed axes.
    """
    return subband_keys(dimensions)


def _spatial_axes(dimensions: int) -> tuple[int, ...]:
    """Trailing axes a `dimensions`-dimensional transform runs over.

    Args:
        dimensions: Number of spatial dimensions.
    """
    return tuple(range(-dimensions, 0))


def _check_rank(x: torch.Tensor, dimensions: int) -> None:
    """Check that a tensor carries a batch and channel axis plus the spatial ones.

    Args:
        x: Input tensor.
        dimensions: Number of spatial dimensions.

    Raises:
        ValueError: If the tensor rank does not match `dimensions`.
    """
    if x.dim() != dimensions + 2:
        raise ValueError(
            f"Expected a {dimensions + 2}-dimensional input for {dimensions} spatial"
            f" dimension(s); got an input of shape {tuple(x.shape)}."
        )


def _stack(
    bands: Sequence[torch.Tensor], order: SubbandOrderTypes
) -> torch.Tensor:
    """Fold the subbands into the channel axis.

    Args:
        bands: Subbands, in the order `subband_names` gives them.
        order: `'subband'` groups by subband, `'channel'` groups by input channel.

    Raises:
        ValueError: If `order` is not a known subband order.
    """
    match order:
        case "subband":
            return torch.cat(list(bands), dim=1)
        case "channel":
            return torch.stack(list(bands), dim=2).flatten(1, 2)
        case _:
            raise ValueError(
                f"Unsupported subband order: {order!r}. Use 'subband' or 'channel'."
            )


def _unstack(
    x: torch.Tensor, dimensions: int, order: SubbandOrderTypes
) -> tuple[torch.Tensor, ...]:
    """Split the channel axis back into subbands.

    Args:
        x: Stacked subbands.
        dimensions: Number of spatial dimensions.
        order: Subband order the channel axis carries.

    Raises:
        ValueError: If the channel count is not a multiple of the subband count,
            or if `order` is not a known subband order.
    """
    count = 2**dimensions
    if x.shape[1] % count:
        raise ValueError(
            f"A {dimensions}-dimensional transform carries {count} subbands, so the"
            f" channel count must be a multiple of {count}; got {x.shape[1]}."
        )
    match order:
        case "subband":
            return x.chunk(count, dim=1)
        case "channel":
            return x.unflatten(1, (x.shape[1] // count, count)).unbind(dim=2)
        case _:
            raise ValueError(
                f"Unsupported subband order: {order!r}. Use 'subband' or 'channel'."
            )


def dwt_nd(
    x: torch.Tensor,
    dimensions: int,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    subband_order: SubbandOrderTypes = "subband",
) -> torch.Tensor:
    """Single-level transform of a batch, stacking the subbands on the channels.

    Args:
        x: Input tensor, shaped `(N, C, *spatial)`.
        dimensions: Number of spatial dimensions.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        subband_order: How the subbands are laid out in the channel axis.
    """
    _check_rank(x, dimensions)
    bands = dwtn(x, wavelet, mode, _spatial_axes(dimensions))
    return _stack([bands[key] for key in subband_keys(dimensions)], subband_order)


def idwt_nd(
    x: torch.Tensor,
    dimensions: int,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    subband_order: SubbandOrderTypes = "subband",
    output_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Invert a single-level transform whose subbands sit in the channel axis.

    Args:
        x: Stacked subbands, shaped `(N, 2**dimensions * C, *spatial)`.
        dimensions: Number of spatial dimensions.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode the analysis used.
        subband_order: How the subbands are laid out in the channel axis.
        output_size: Length of each spatial axis in the reconstruction.
    """
    _check_rank(x, dimensions)
    parts = _unstack(x, dimensions, subband_order)
    coeffs = dict(zip(subband_keys(dimensions), parts, strict=True))
    return idwtn(coeffs, wavelet, mode, _spatial_axes(dimensions), output_size)


def dwt_nd_approx(
    x: torch.Tensor,
    dimensions: int,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
) -> torch.Tensor:
    """Approximation band of a single-level transform, leaving the channels alone.

    Args:
        x: Input tensor, shaped `(N, C, *spatial)`.
        dimensions: Number of spatial dimensions.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
    """
    _check_rank(x, dimensions)
    return dwtn_approx(x, wavelet, mode, _spatial_axes(dimensions))


def wavedec_nd(
    x: torch.Tensor,
    dimensions: int,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    levels: int = 1,
    subband_order: SubbandOrderTypes = "subband",
) -> list[torch.Tensor]:
    """Transform a batch repeatedly, recursing on the approximation band.

    Args:
        x: Input tensor, shaped `(N, C, *spatial)`.
        dimensions: Number of spatial dimensions.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode.
        levels: Number of levels.
        subband_order: How the subbands are laid out in the channel axis.

    Returns:
        One stacked tensor per level, finest first; level `l` holds every
        subband of that level at `1 / 2**l` of the input resolution.

    Raises:
        ValueError: If `levels` is not positive.
    """
    if levels < 1:
        raise ValueError(f"A decomposition needs at least one level; got {levels}.")
    out, approx = [], x
    for _ in range(levels):
        stacked = dwt_nd(approx, dimensions, wavelet, mode, subband_order)
        out.append(stacked)
        approx = _unstack(stacked, dimensions, subband_order)[0]
    return out


def waverec_nd(
    coeffs: Sequence[torch.Tensor],
    dimensions: int,
    wavelet: str | Wavelet = "haar",
    mode: ExtensionModeTypes = "zero",
    subband_order: SubbandOrderTypes = "subband",
    output_size: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Invert a decomposition produced by `wavedec_nd`.

    Args:
        coeffs: One stacked tensor per level, finest first.
        dimensions: Number of spatial dimensions.
        wavelet: Wavelet, by name or as a `Wavelet`.
        mode: Signal extension mode the analysis used.
        subband_order: How the subbands are laid out in the channel axis.
        output_size: Per level, the spatial shape to reconstruct, coarsest first.

    Raises:
        ValueError: If `coeffs` is empty.
    """
    if not coeffs:
        raise ValueError("`coeffs` must hold at least one level.")
    approx = None
    for i, level in enumerate(reversed(coeffs)):
        if approx is not None:
            parts = list(_unstack(level, dimensions, subband_order))
            parts[0] = approx
            level = _stack(parts, subband_order)
        sizes = None if output_size is None else output_size[i]
        approx = idwt_nd(level, dimensions, wavelet, mode, subband_order, sizes)
    return approx

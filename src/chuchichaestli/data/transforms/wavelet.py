# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Discrete wavelet transforms over the trailing spatial dimensions."""

import warnings
from collections.abc import Sequence
from typing import Any

import torch
from torchvision.transforms.v2 import Transform

from chuchichaestli.dwt.functional import dwt_coeff_len, dwtn, idwtn, subband_keys
from chuchichaestli.dwt.modes import ExtensionModeTypes
from chuchichaestli.dwt.wavelet import Wavelet, wavelet


__all__ = [
    "WaveletDecompose",
    "InvWaveletDecompose",
    "WaveletPacket",
    "InvWaveletPacket",
]


class WaveletDecompose(Transform):
    """Replace the trailing `ndim` axes by the subbands of a wavelet transform.

    The subbands are stacked on one new axis, so the result stays a single
    tensor and collates like any other field. Leading axes pass through. Only
    one level is taken: recursing on the coarsest subband alone would leave the
    levels at different resolutions, which no single tensor can hold. See
    `WaveletPacket` for the deeper decomposition that does keep one resolution.

    Reverting needs the spatial shape the coefficients came from, which the
    transform records as it runs; pass `sizes` to fix it up front instead.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        wavelet: str | Wavelet = "haar",
        ndim: int = 2,
        mode: ExtensionModeTypes = "zero",
        stack_dim: int | None = None,
        sizes: Sequence[int] | None = None,
    ):
        """Constructor.

        Args:
            wavelet: Wavelet to decompose with, by name or as a `Wavelet`.
            ndim: Number of trailing spatial dims to transform.
            mode: Signal extension mode; `'periodization'` is the only one that
                halves each axis exactly, and so the only one that does not grow
                the data.
            stack_dim: Position of the subband axis in the result; immediately
                before the spatial axes if omitted.
            sizes: Spatial shape the coefficients revert to, if it is known.

        Raises:
            ValueError: If `ndim` is out of range.
        """
        super().__init__()
        if not 1 <= ndim <= 3:
            raise ValueError(f"The transform runs over one to three axes; got {ndim}.")
        levels = 1
        self.wavelet = wavelet
        self.ndim = ndim
        self.mode = mode
        self.levels = levels
        self.stack_dim = -ndim - 1 if stack_dim is None else stack_dim
        self.sizes = None if sizes is None else tuple(int(n) for n in sizes)
        self._declared = sizes is not None
        self._coerced: Wavelet | None = None

    @property
    def axes(self) -> tuple[int, ...]:
        """Axes the transform runs over, as trailing indices."""
        return tuple(range(-self.ndim, 0))

    @property
    def subbands(self) -> int:
        """Number of subbands the transform produces."""
        return 2**self.ndim

    def _wavelet(self) -> Wavelet:
        """Return the wavelet, coercing a name once and keeping its filter cache."""
        if self._coerced is None:
            self._coerced = wavelet(self.wavelet)
        return self._coerced

    def __getstate__(self) -> dict:
        """Drop the coerced wavelet so workers rebuild its filters lazily."""
        return {k: v for k, v in self.__dict__.items() if k != "_coerced"}

    def __setstate__(self, state: dict) -> None:
        """Restore the transform without a coerced wavelet."""
        self.__dict__.update(state)
        self._coerced = None

    def _record(self, shape: tuple[int, ...]) -> None:
        """Record the spatial shape being decomposed, complaining if it changes.

        Args:
            shape: Trailing spatial shape of the input.

        Raises:
            ValueError: If a declared shape does not match the input.
        """
        if self.sizes == shape:
            return
        if self.sizes is not None:
            if self._declared:
                raise ValueError(
                    f"the transform was declared to revert to {self.sizes} but the"
                    f" input has shape {shape}; the two cannot both be reverted"
                )
            warnings.warn(
                f"the transform previously ran on {self.sizes} and is now given"
                f" {shape}; coefficients produced under the old shape can no"
                " longer be reverted by this instance. Pass `sizes` explicitly,"
                " or use one instance per shape.",
                UserWarning,
            )
        self.sizes = shape

    def _ladder(self, shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Spatial shape entering each level, finest first.

        Args:
            shape: Trailing spatial shape of the input.
        """
        filter_len = self._wavelet().dec_len
        ladder, current = [], list(shape)
        for _ in range(self.levels):
            ladder.append(tuple(current))
            current = [dwt_coeff_len(n, filter_len, self.mode) for n in current]
        return ladder

    def _decompose(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose and stack the subbands onto one axis.

        Args:
            x: Input tensor.
        """
        wave = self._wavelet()
        bands = [x]
        for _ in range(self.levels):
            bands = [
                decomposed[key]
                for band in bands
                for decomposed in (dwtn(band, wave, self.mode, self.axes),)
                for key in subband_keys(self.ndim)
            ]
        return torch.stack(bands, dim=self.stack_dim)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the stacked subbands back into the original axes.

        Args:
            x: Stacked subbands.

        Raises:
            ValueError: If the spatial shape to revert to is unknown, or if the
                subband axis does not hold the expected number of bands.
        """
        if self.sizes is None:
            raise ValueError(
                "the spatial shape to revert to is unknown; run `transform` first"
                " or pass `sizes` to the constructor"
            )
        expected = self.subbands**self.levels
        if x.shape[self.stack_dim] != expected:
            raise ValueError(
                f"a {self.levels}-level transform over {self.ndim} axes stacks"
                f" {expected} subbands; got {x.shape[self.stack_dim]}."
            )
        wave = self._wavelet()
        keys = subband_keys(self.ndim)
        bands = list(x.unbind(dim=self.stack_dim))
        for shape in reversed(self._ladder(self.sizes)):
            bands = [
                idwtn(
                    dict(zip(keys, bands[i : i + len(keys)], strict=True)),
                    wave,
                    self.mode,
                    self.axes,
                    shape,
                )
                for i in range(0, len(bands), len(keys))
            ]
        return bands[0]

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Record the spatial shape of the first tensor, shared by every leaf."""
        x = next(i for i in flat_inputs if isinstance(i, torch.Tensor))
        shape = tuple(int(n) for n in x.shape[-self.ndim :])
        self._record(shape)
        return {"sizes": shape}

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Decompose the trailing spatial axes."""
        self._record(tuple(int(n) for n in x.shape[-self.ndim :]))
        return self._decompose(x)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Reconstruct the trailing spatial axes."""
        return self._reconstruct(x)

    def get_inverse(self) -> "InvWaveletDecompose":
        """Return the transform that reconstructs what this one decomposes."""
        return InvWaveletDecompose(
            self.wavelet, self.ndim, self.mode, self.stack_dim, self.sizes
        )


class InvWaveletDecompose(WaveletDecompose):
    """Inverse of `WaveletDecompose`: reconstruct forward, decompose on revert."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Reconstruct the trailing spatial axes."""
        return self._reconstruct(x)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Decompose the trailing spatial axes."""
        self._record(tuple(int(n) for n in x.shape[-self.ndim :]))
        return self._decompose(x)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Report the recorded spatial shape; the forward pass reads, not writes."""
        return {"sizes": self.sizes}

    def get_inverse(self) -> "WaveletDecompose":
        """Return the transform that decomposes what this one reconstructs."""
        return WaveletDecompose(
            self.wavelet, self.ndim, self.mode, self.stack_dim, self.sizes
        )


class WaveletPacket(WaveletDecompose):
    """`WaveletDecompose` that keeps splitting every subband, not only the coarsest.

    A packet decomposition of `l` levels leaves `2 ** (ndim * l)` subbands, all
    at the same resolution, which is what makes the result a single tensor.
    """

    def __init__(
        self,
        wavelet: str | Wavelet = "haar",
        ndim: int = 2,
        mode: ExtensionModeTypes = "zero",
        levels: int = 2,
        stack_dim: int | None = None,
        sizes: Sequence[int] | None = None,
    ):
        """Constructor.

        Args:
            wavelet: Wavelet to decompose with, by name or as a `Wavelet`.
            ndim: Number of trailing spatial dims to transform.
            mode: Signal extension mode.
            levels: Number of levels to decompose.
            stack_dim: Position of the subband axis in the result.
            sizes: Spatial shape the coefficients revert to, if it is known.

        Raises:
            ValueError: If `levels` is not positive.
        """
        super().__init__(wavelet, ndim, mode, stack_dim, sizes)
        if levels < 1:
            raise ValueError(f"A decomposition needs at least one level; got {levels}.")
        self.levels = levels

    def get_inverse(self) -> "InvWaveletPacket":
        """Return the transform that reconstructs what this one decomposes."""
        return InvWaveletPacket(
            self.wavelet,
            self.ndim,
            self.mode,
            self.levels,
            self.stack_dim,
            self.sizes,
        )


class InvWaveletPacket(WaveletPacket, InvWaveletDecompose):
    """Inverse of `WaveletPacket`: reconstruct forward, decompose on revert.

    Takes the level count from `WaveletPacket` and the swapped direction from
    `InvWaveletDecompose`, in that order.
    """

    def get_inverse(self) -> "WaveletPacket":
        """Return the transform that decomposes what this one reconstructs."""
        return WaveletPacket(
            self.wavelet,
            self.ndim,
            self.mode,
            self.levels,
            self.stack_dim,
            self.sizes,
        )

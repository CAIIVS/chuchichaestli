# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wavelet filter banks and the registry of the wavelets shipped with chuchichaestli."""

import torch

from chuchichaestli.dwt.filters import (
    ALIASES,
    BIORTHOGONAL_BANKS,
    FAMILIES,
    ORTHOGONAL_DEC_LO,
)
from collections.abc import Sequence
from functools import cache
from typing import Literal


__all__ = ["Wavelet", "WaveletTypes", "WAVELET_REGISTRY", "wavelet", "wavelist"]


WaveletTypes = Literal[
    "haar",
    "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8",
    "sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8",
    "coif1", "coif2", "coif3",
    "bior1.1", "bior1.3", "bior1.5",
    "bior2.2", "bior2.4", "bior2.6", "bior2.8",
    "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9",
    "bior4.4",
    "rbio1.1", "rbio1.3", "rbio1.5",
    "rbio2.2", "rbio2.4", "rbio2.6", "rbio2.8",
    "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", "rbio3.9",
    "rbio4.4",
]

# name -> family, for every wavelet the registry can build by name
WAVELET_REGISTRY: dict[str, str] = {
    **{name: family for family, names in FAMILIES.items() for name in names},
    **dict.fromkeys(ALIASES, "db"),
}


def _mirror(filt: Sequence[float]) -> tuple[float, ...]:
    """Alternate the sign of every other coefficient (modulation by `(-1)**k`).

    Args:
        filt: Filter coefficients.
    """
    return tuple((-1.0) ** k * v for k, v in enumerate(filt))


def _neg_mirror(filt: Sequence[float]) -> tuple[float, ...]:
    """Alternate the sign of every other coefficient, starting with a negation.

    Args:
        filt: Filter coefficients.
    """
    return tuple((-1.0) ** (k + 1) * v for k, v in enumerate(filt))


def _convolve(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Convolve two short coefficient sequences.

    Args:
        a: First sequence.
        b: Second sequence.
    """
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Add two coefficient sequences, zero-padding the shorter one.

    Args:
        a: First sequence.
        b: Second sequence.
    """
    n = max(len(a), len(b))
    return [
        (a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0)
        for i in range(n)
    ]


class Wavelet:
    """A two-channel wavelet filter bank.

    A bank pairs a decomposition (analysis) and a reconstruction (synthesis)
    low- and high-pass filter. Only the low-pass filters are independent, so
    `__init__` derives whatever is not given.

    Attributes:
        name: Name of the wavelet.
        family: Family the wavelet belongs to (`'db'`, `'sym'`, ...).
        dec_lo: Decomposition low-pass filter.
        dec_hi: Decomposition high-pass filter.
        rec_lo: Reconstruction low-pass filter.
        rec_hi: Reconstruction high-pass filter.
        orthogonal: Whether the bank is orthogonal.
        biorthogonal: Whether the bank is biorthogonal (orthogonal banks are).
    """

    def __init__(
        self,
        dec_lo: Sequence[float],
        dec_hi: Sequence[float] | None = None,
        rec_lo: Sequence[float] | None = None,
        rec_hi: Sequence[float] | None = None,
        *,
        name: str = "custom",
        family: str = "custom",
        orthogonal: bool | None = None,
        biorthogonal: bool | None = None,
    ):
        """Constructor.

        Args:
            dec_lo: Decomposition low-pass filter.
            dec_hi: Decomposition high-pass filter; derived from `rec_lo` if omitted.
            rec_lo: Reconstruction low-pass filter; the reversed `dec_lo` if
                omitted, which makes the bank orthogonal.
            rec_hi: Reconstruction high-pass filter; derived from `dec_lo` if omitted.
            name: Name of the wavelet.
            family: Family the wavelet belongs to.
            orthogonal: Whether the bank is orthogonal; inferred if omitted.
            biorthogonal: Whether the bank is biorthogonal; inferred if omitted.

        Raises:
            ValueError: If a filter is empty, if the filters differ in length, or
                if a bank declared orthogonal does not reconstruct perfectly.
        """
        if not len(dec_lo):
            raise ValueError("A wavelet needs at least one filter coefficient.")
        derived = rec_lo is None
        dec_lo = tuple(float(v) for v in dec_lo)
        rec_lo = tuple(float(v) for v in rec_lo) if rec_lo is not None else dec_lo[::-1]
        rec_hi = tuple(rec_hi) if rec_hi is not None else _mirror(dec_lo)
        dec_hi = tuple(dec_hi) if dec_hi is not None else _neg_mirror(rec_lo)

        self.name = name
        self.family = family
        self.dec_lo = dec_lo
        self.dec_hi = tuple(float(v) for v in dec_hi)
        self.rec_lo = rec_lo
        self.rec_hi = tuple(float(v) for v in rec_hi)

        lengths = {len(f) for f in self.filter_bank}
        if len(lengths) != 1:
            raise ValueError(
                f"All four filters of a wavelet must have the same length; got"
                f" {[len(f) for f in self.filter_bank]}."
            )

        self.orthogonal = derived if orthogonal is None else bool(orthogonal)
        self.biorthogonal = (
            self.orthogonal or self.check_perfect_reconstruction()
            if biorthogonal is None
            else bool(biorthogonal)
        )
        if self.orthogonal and not self.check_perfect_reconstruction():
            raise ValueError(
                f"The filter bank of {self.name!r} was declared orthogonal but does"
                f" not reconstruct perfectly."
            )
        self._cache: dict[tuple[str, str], tuple[torch.Tensor, ...]] = {}

    @property
    def filter_bank(self) -> tuple[tuple[float, ...], ...]:
        """The four filters, in the order `(dec_lo, dec_hi, rec_lo, rec_hi)`."""
        return (self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi)

    @property
    def dec_len(self) -> int:
        """Length of the decomposition filters."""
        return len(self.dec_lo)

    @property
    def rec_len(self) -> int:
        """Length of the reconstruction filters."""
        return len(self.rec_lo)

    @property
    def filter_len(self) -> int:
        """Length of the filters; decomposition and reconstruction share it."""
        return len(self.dec_lo)

    @classmethod
    @cache
    def from_name(cls, name: str) -> "Wavelet":
        """Build one of the available wavelets.

        Cached: deriving a bank and checking that it reconstructs costs more
        than a small transform does, and the result only depends on the name.
        The filter tensors a caller asks for are cached on the instance, so
        sharing it also shares those.

        Args:
            name: Name of the wavelet, e.g. `'haar'`, `'db4'` or `'bior2.2'`.

        Raises:
            ValueError: If `name` is not a known wavelet.
        """
        key = ALIASES.get(name, name)
        if key not in WAVELET_REGISTRY:
            raise ValueError(
                f"Unknown wavelet: {name!r}. Use one of {sorted(WAVELET_REGISTRY)},"
                f" or pass an explicit filter bank to `Wavelet`."
            )
        family = WAVELET_REGISTRY[key]
        if key in ORTHOGONAL_DEC_LO:
            return cls(
                ORTHOGONAL_DEC_LO[key], name=name, family=family, orthogonal=True
            )
        if family == "rbio":
            # A reverse-biorthogonal bank is its biorthogonal partner with the
            # analysis and synthesis sides swapped and every filter reversed.
            partner = cls.from_name(key.replace("rbio", "bior"))
            return cls(
                partner.rec_lo[::-1],
                partner.rec_hi[::-1],
                partner.dec_lo[::-1],
                partner.dec_hi[::-1],
                name=name,
                family=family,
                orthogonal=False,
                biorthogonal=True,
            )
        dec_lo, rec_lo = BIORTHOGONAL_BANKS[key]
        return cls(
            dec_lo,
            rec_lo=rec_lo,
            name=name,
            family=family,
            orthogonal=False,
            biorthogonal=True,
        )

    def filters(
        self, dtype: torch.dtype, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the filter bank as tensors, cached per dtype and device.

        Instances are shared, so the cached tensors are handed out detached:
        a caller marking them for a gradient, or otherwise touching them in
        place, then changes only its own view of them.

        Args:
            dtype: Floating point type of the returned tensors.
            device: Device the returned tensors live on.
        """
        key = (str(dtype), str(device))
        if key not in self._cache:
            self._cache[key] = tuple(
                torch.tensor(f, dtype=dtype, device=device) for f in self.filter_bank
            )
        return tuple(filt.detach() for filt in self._cache[key])

    def check_perfect_reconstruction(self, atol: float = 1e-10) -> bool:
        """Check that the bank introduces no distortion and cancels its aliasing.

        Args:
            atol: Absolute tolerance the two conditions are checked to.
        """
        distortion = _add(
            _convolve(self.rec_lo, self.dec_lo), _convolve(self.rec_hi, self.dec_hi)
        )
        alias = _add(
            _convolve(self.rec_lo, _mirror(self.dec_lo)),
            _convolve(self.rec_hi, _mirror(self.dec_hi)),
        )
        delay = self.dec_len - 1
        expected = [2.0 if n == delay else 0.0 for n in range(len(distortion))]
        return all(abs(a - b) <= atol for a, b in zip(distortion, expected)) and all(
            abs(v) <= atol for v in alias
        )

    def __repr__(self) -> str:
        """Representation of the wavelet."""
        kind = "orthogonal" if self.orthogonal else "biorthogonal"
        return (
            f"{type(self).__name__}(name={self.name!r}, family={self.family!r},"
            f" filter_len={self.filter_len}, {kind}=True)"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two wavelets by name and filter bank."""
        if not isinstance(other, Wavelet):
            return NotImplemented
        return self.name == other.name and self.filter_bank == other.filter_bank

    def __hash__(self) -> int:
        """Hash the wavelet by name and filter bank."""
        return hash((self.name, self.filter_bank))

    def __getstate__(self) -> dict:
        """Drop the tensor cache so workers rebuild it on their own device."""
        return {k: v for k, v in self.__dict__.items() if k != "_cache"}

    def __setstate__(self, state: dict) -> None:
        """Restore the wavelet with an empty tensor cache."""
        self.__dict__.update(state)
        self._cache = {}


def wavelist(family: str | None = None) -> list[str]:
    """List the available wavelets.

    Args:
        family: Restrict the listing to one family; all families if omitted.

    Raises:
        ValueError: If `family` is not a known family.
    """
    if family is None:
        return sorted(WAVELET_REGISTRY)
    if family not in FAMILIES:
        raise ValueError(
            f"Unknown wavelet family: {family!r}. Use one of {sorted(FAMILIES)}."
        )
    return list(FAMILIES[family])


def wavelet(spec: "str | Wavelet") -> Wavelet:
    """Coerce a wavelet name into a `Wavelet`, passing instances through.

    Args:
        spec: Name of a shipped wavelet, or a `Wavelet` to use as is.
    """
    return spec if isinstance(spec, Wavelet) else Wavelet.from_name(spec)

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Separable projection onto truncated tensor-product bases."""

import math
import warnings
from collections.abc import Callable
from typing import Any
import torch
from torchvision.transforms.v2 import Transform

__all__ = ["BasisProjection", "InvBasisProjection", "BASIS_REGISTRY"]


def _chebyshev(n: int, m: int) -> torch.Tensor:
    """Chebyshev polynomials of the first kind on `n` equispaced nodes."""
    x = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    k = torch.arange(m, dtype=torch.float64)
    return torch.cos(k * torch.arccos(x.clamp(-1.0, 1.0)).unsqueeze(1))


def _legendre(n: int, m: int) -> torch.Tensor:
    """Legendre polynomials on `n` equispaced nodes, via the recurrence."""
    x = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    cols = [torch.ones_like(x)]
    if m > 1:
        cols.append(x)
    for deg in range(1, m - 1):
        # (deg+1) P_{deg+1} = (2 deg + 1) x P_deg - deg P_{deg-1}
        cols.append(((2 * deg + 1) * x * cols[deg] - deg * cols[deg - 1]) / (deg + 1))
    return torch.stack(cols[:m], dim=1)


def _monomial(n: int, m: int) -> torch.Tensor:
    """Raw monomials on `n` equispaced nodes (ill-conditioned; for reference)."""
    x = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    return torch.stack([x**k for k in range(m)], dim=1)


def _fourier(n: int, m: int) -> torch.Tensor:
    """Real Fourier series: `[1, cos, sin, cos2, sin2, ...]` truncated to `m`."""
    x = torch.arange(n, dtype=torch.float64) / n
    cols = [torch.ones_like(x)]
    harmonic = 1
    while len(cols) < m:
        cols.append(torch.cos(2 * math.pi * harmonic * x))
        if len(cols) < m:
            cols.append(torch.sin(2 * math.pi * harmonic * x))
        harmonic += 1
    return torch.stack(cols[:m], dim=1)


def _dct(n: int, m: int) -> torch.Tensor:
    """DCT-II basis, orthogonal on the `n`-point grid."""
    j = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    k = torch.arange(m, dtype=torch.float64)
    return torch.cos(math.pi * (j + 0.5) * k / n)


BASIS_REGISTRY: dict[str, Callable[[int, int], torch.Tensor]] = {
    "chebyshev": _chebyshev,
    "legendre": _legendre,
    "monomial": _monomial,
    "fourier": _fourier,
    "dct": _dct,
}


class BasisProjection(Transform):
    """Project selected axes onto truncated bases, and reconstruct back.

    The forward pass replaces each selected axis of length `N` by `M`
    coefficients of a truncated expansion, `revert` reconstructs the original
    length. With a design matrix `B_k` per axis this is a decomposition
    with fixed (non-learned) factor matrices:

        C = X x_1 B_1^+ x_2 B_2^+ ...    (project)
        X = C x_1 B_1   x_2 B_2   ...    (reconstruct)

    Complex input is supported: the real design matrices are promoted to the
    input dtype, so real and imaginary parts project independently.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        bases: dict[int, tuple[str, int] | int | torch.Tensor],
        basis: str = "chebyshev",
        weights: dict[int, torch.Tensor] | None = None,
        rcond: float | None = None,
        lengths: dict[int, int] | None = None,
        cond_warn: float | None = 1e6,
    ):
        """Constructor.

        Args:
            bases: Maps an axis of the input to its basis. Each value is either
                an `(name, order)` pair, a bare `order` using the default
                `basis` family, or an explicit `(N, M)` design matrix (e.g.
                fitted PCA components). Axes may be negative.
            basis: Default family for entries given as a bare order. One of
                `BASIS_REGISTRY`: chebyshev, legendre, monomial, fourier, dct.
            weights: Optional per-axis 1-D non-negative weights of length
                `N` for a weighted least-squares fit, e.g. for measurement
                noise or masked samples. Axes as in `bases`, sign included.
            rcond: Relative cutoff for small singular values in the
                pseudoinverse, forwarded to `torch.linalg.pinv` as `rtol`.
            lengths: Original length `N` per axis, which coefficients do not
                carry. Projecting records it, but pass it explicitly to revert
                standalone or across processes. Axes as in `bases`, sign
                included.
            cond_warn: Warn when the fitted matrix's condition number exceeds
                this. `None` disables the check.
        """
        super().__init__()
        if not bases:
            raise ValueError("bases must name at least one axis")
        normalised: dict[int, tuple[str, int] | torch.Tensor] = {}
        for axis, spec in bases.items():
            if isinstance(spec, torch.Tensor):
                if spec.ndim != 2:
                    raise ValueError(
                        f"design matrix for axis {axis} must be 2-D (N, M), "
                        f"got shape {tuple(spec.shape)}"
                    )
                normalised[axis] = spec
                continue
            name, order = (basis, spec) if isinstance(spec, int) else spec
            if name not in BASIS_REGISTRY:
                raise ValueError(
                    f"unknown basis {name!r}; choose from {sorted(BASIS_REGISTRY)}"
                )
            if order < 1:
                raise ValueError(f"order for axis {axis} must be >= 1, got {order}")
            normalised[axis] = (name, order)
        for name, mapping in (("weights", weights), ("lengths", lengths)):
            stray = sorted(set(mapping or ()) - set(normalised))
            if stray:
                raise ValueError(
                    f"{name} names axes {stray} that are absent from bases "
                    f"{sorted(normalised)}; write them with the same sign"
                )
        self.bases = normalised
        self.basis = basis
        self.weights = {a: w.detach().clone() for a, w in (weights or {}).items()}
        self.rcond = rcond
        self.cond_warn = cond_warn
        self.lengths = dict(lengths) if lengths else {}  # original axis lengths
        self._declared = set(self.lengths)
        for axis, spec in normalised.items():
            if isinstance(spec, torch.Tensor):
                self.lengths.setdefault(axis, spec.shape[0])
                self._declared.add(axis)
        self._cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}

    def __getstate__(self) -> dict:
        """Drop the matrix cache so workers rebuild it lazily."""
        return {k: v for k, v in self.__dict__.items() if k != "_cache"}

    def __setstate__(self, state: dict) -> None:
        """Restore state with an empty cache."""
        self.__dict__.update(state)
        self._cache = {}

    def design_matrix(self, axis: int, n: int) -> torch.Tensor:
        """Return the `(n, M)` design matrix for `axis`, in float64.

        Built-in families are built on the CPU; an explicit design matrix is
        returned on whichever device the caller supplied it on.
        """
        spec = self.bases[axis]
        if isinstance(spec, torch.Tensor):
            if spec.shape[0] != n:
                raise ValueError(
                    f"design matrix for axis {axis} has {spec.shape[0]} rows but "
                    f"the input has length {n} along that axis"
                )
            return spec.to(torch.float64)
        name, order = spec
        if order > n:
            raise ValueError(
                f"order {order} exceeds the length {n} of axis {axis}; the fit "
                "would be underdetermined"
            )
        return BASIS_REGISTRY[name](n, order)

    def _check_conditioning(self, axis: int, basis: torch.Tensor) -> None:
        """Warn when the fitted matrix is too ill-conditioned to fit stably.

        `basis` is the matrix actually passed to the pseudoinverse, so for a
        weighted fit it is the whitened one.
        """
        if self.cond_warn is None:
            return
        svals = torch.linalg.svdvals(basis)
        smallest = svals[-1]
        cond = float("inf") if smallest <= 0 else float(svals[0] / smallest)
        if cond > self.cond_warn:
            n, order = basis.shape
            warnings.warn(
                f"axis {axis}: design matrix is ill-conditioned (cond={cond:.2e}) "
                f"for {order} terms on {n} samples. Least-squares fitting from "
                f"equispaced samples is stable only up to about "
                f"2*sqrt(N)={2 * math.sqrt(n):.1f} terms. Reduce the order, or "
                "set `rcond` to truncate small singular values.",
                UserWarning,
            )

    def _matrices(
        self, axis: int, n: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(B, B_pinv)` for `axis`, cast to `dtype` on `device`."""
        key = (axis, n, str(dtype), str(device))
        if key in self._cache:
            return self._cache[key]

        weight = self.weights.get(axis)
        basis = self.design_matrix(axis, n)
        kw = {} if self.rcond is None else {"rtol": self.rcond}
        if weight is None:
            self._check_conditioning(axis, basis)
            pinv = torch.linalg.pinv(basis, **kw)
        else:
            w = weight.to(device=basis.device, dtype=torch.float64).reshape(-1)
            if w.numel() != n:
                raise ValueError(
                    f"weights for axis {axis} have length {w.numel()}, expected {n}"
                )
            if bool((w < 0).any()):
                raise ValueError(
                    f"weights for axis {axis} must be non-negative, got a "
                    f"minimum of {float(w.min())}"
                )
            # Weighted least squares using whitened matrix for better accuracy
            root = w.sqrt().unsqueeze(1)
            whitened = basis * root
            self._check_conditioning(axis, whitened)
            pinv = torch.linalg.pinv(whitened, **kw) * root.reshape(1, -1)

        basis, pinv = basis.to(dtype).to(device), pinv.to(dtype).to(device)
        self._cache[key] = (basis, pinv)
        return basis, pinv

    @staticmethod
    def _apply_along(x: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
        """Contract `x` with `matrix` along `axis` (matrix is applied on the left)."""
        moved = x.movedim(axis, -1)
        out = moved @ matrix.transpose(-2, -1)
        return out.movedim(-1, axis)

    def _resolve(self, ndim: int) -> list[tuple[int, int]]:
        """Return `(key, positive_axis)` pairs, rejecting duplicate axes."""
        resolved = []
        seen: dict[int, int] = {}
        for key in sorted(self.bases):
            # Mirror tensor indexing: only [-ndim, ndim) is addressable, so a
            # further-negative axis is an error rather than a wrap-around.
            if not -ndim <= key < ndim:
                raise ValueError(
                    f"axis {key} is out of range for a {ndim}-dimensional input"
                )
            axis = key % ndim if key < 0 else key
            if axis in seen:
                raise ValueError(
                    f"axes {seen[axis]} and {key} both refer to axis {axis}"
                )
            seen[axis] = key
            resolved.append((key, axis))
        return resolved

    def _record_length(self, key: int, n: int) -> None:
        """Record the projected length of `key`, complaining if it changes."""
        known = self.lengths.get(key)
        if known == n:
            return
        if known is not None:
            if key in self._declared:
                raise ValueError(
                    f"axis {key} was declared to have length {known} but the "
                    f"input has length {n}; the two cannot both be reverted"
                )
            warnings.warn(
                f"axis {key} was previously projected at length {known} and is "
                f"now {n}; coefficients produced under the old length can no "
                "longer be reverted by this instance. Pass `lengths` "
                "explicitly, or use one instance per length.",
                UserWarning,
            )
        self.lengths[key] = n

    @staticmethod
    def _as_inexact(x: torch.Tensor) -> torch.Tensor:
        """Promote integer or boolean input, which cannot carry coefficients."""
        if x.dtype.is_floating_point or x.dtype.is_complex:
            return x
        return x.to(torch.float32)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Replace each selected axis by its coefficients."""
        x = self._as_inexact(x)
        for key, axis in self._resolve(x.ndim):
            n = x.shape[axis]
            self._record_length(key, n)
            _, pinv = self._matrices(key, n, x.dtype, x.device)
            x = self._apply_along(x, pinv, axis)
        return x

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Expand each selected axis back from its coefficients."""
        x = self._as_inexact(x)
        for key, axis in self._resolve(x.ndim):
            spec = self.bases[key]
            order = spec.shape[1] if isinstance(spec, torch.Tensor) else spec[1]
            if x.shape[axis] != order:
                raise ValueError(
                    f"axis {key} has length {x.shape[axis]} but the basis holds "
                    f"{order} coefficients"
                )
            if key not in self.lengths:
                raise RuntimeError(
                    f"the original length of axis {key} is unknown; pass "
                    "`lengths`, supply an explicit design matrix, or project "
                    "with this instance first"
                )
            basis, _ = self._matrices(key, self.lengths[key], x.dtype, x.device)
            x = self._apply_along(x, basis, axis)
        return x

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Project onto the truncated bases (ignores `params`)."""
        return self._project(x)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Reconstruct from the coefficients (ignores `params`)."""
        return self._reconstruct(x)


class InvBasisProjection(BasisProjection):
    """Inverse of `BasisProjection`: reconstruct forward, project on revert.

    The original axis lengths cannot be inferred from coefficients alone, so
    every axis needs one of: an entry in `lengths`, an explicit design matrix
    (whose row count supplies it), or a prior projection through this instance.
    """

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Reconstruct from the coefficients."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Project onto the truncated bases."""
        return super().transform(x, params)

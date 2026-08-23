# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Value transforms acting on pixel/voxel intensities (stretches and scalers)."""

from typing import Any
from collections.abc import Callable
import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import Transform

__all__ = [
    "Affine",
    "HEPTransform",
    "InvHEPTransform",
    "LogTransform",
    "InvLogTransform",
    "LogP1Transform",
    "InvLogP1Transform",
    "MinMaxScale",
    "InvMinMaxScale",
    "Clamp",
    "HounsfieldScale",
    "InvHounsfieldScale",
    "HounsfieldClamp",
    "ZScaleInterval",
    "ZScale",
]


def _as_tensor(v: Any) -> torch.Tensor:
    """Return `v` as a tensor (leaving existing tensors untouched)."""
    return v if isinstance(v, torch.Tensor) else torch.tensor(v)


class Affine(Transform):
    r"""Affine value transform $(x - b) / a$ (location-scale form)."""

    _transformed_types = (torch.Tensor,)

    def __init__(self, a: float = 1.0, b: float = 0.0):
        """Constructor.

        Args:
            a: The scale, i.e. the divisor (e.g. a data range).
            b: The loc, i.e. the offset subtracted (e.g. a data minimum).
        """
        super().__init__()
        self.a = _as_tensor(a)
        self.b = _as_tensor(b)

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the affine map."""
        return (x - self.b) / self.a

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Invert the affine map."""
        return x * self.a + self.b

    def get_inverse(self) -> Transform:
        """Return the inverse affine map."""
        return Affine(a=1.0 / self.a, b=-self.b / self.a)


class HEPTransform(Transform):
    r"""High-energy-physics stretch for a more Gaussian-like value distribution.

    Computes $\hat{x} = ((x - b) / C)^{1/\gamma}$, where $C$ is roughly the data
    range and $b$ the bias (data minimum).
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, gamma: float = 1.0, b: float = 0.0, C: float = 1.0):
        """Constructor.

        Args:
            gamma: Inverse-power exponent of the stretch.
            b: Bias, i.e. the global data minimum.
            C: A number around the global data range.
        """
        super().__init__()
        self.gamma = _as_tensor(gamma)
        self.b = _as_tensor(b)
        self.C = _as_tensor(C)
        self.inv_gamma = 1.0 / self.gamma
        self.affine = Affine(a=self.C, b=self.b)

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the stretch."""
        return torch.pow(self.affine.transform(x), self.inv_gamma)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Invert the stretch."""
        return self.affine.revert(torch.pow(x, self.gamma))

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return InvHEPTransform(gamma=self.gamma, b=self.b, C=self.C)


class InvHEPTransform(HEPTransform):
    """Inverse of `HEPTransform`."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the inverse stretch."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the forward stretch."""
        return super().transform(x, params)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return HEPTransform(gamma=self.gamma, b=self.b, C=self.C)


class LogTransform(Transform):
    r"""Logarithmic stretch $\hat{x} = \log(x)$ (invertible by exponentiation).

    Pure nonlinearity; compose with `Affine` (e.g. via `SequentialTransform`) to
    normalize the log-space range.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, log_fn: Callable = torch.log):
        """Constructor.

        Args:
            log_fn: Logarithm function fixing the base (default: natural log).
        """
        super().__init__()
        self.fn = log_fn
        self.base = torch.exp(
            torch.log(torch.tensor(10.0)) / log_fn(torch.tensor(10.0))
        )

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the log."""
        return self.fn(x)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Invert the log by exponentiation."""
        return torch.pow(self.base, x)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return InvLogTransform(log_fn=self.fn)


class InvLogTransform(LogTransform):
    """Inverse of `LogTransform`."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the exponentiation."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the log."""
        return super().transform(x, params)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return LogTransform(log_fn=self.fn)


class LogP1Transform(LogTransform):
    r"""Log1p-style stretch $\hat{x} = \log(x + 1)$ (handles zeros)."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the log1p stretch."""
        return super().transform(x + 1, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Invert the log1p stretch."""
        return super().revert(x, params) - 1

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return InvLogP1Transform(log_fn=self.fn)


class InvLogP1Transform(LogP1Transform):
    """Inverse of `LogP1Transform`."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the inverse log1p stretch."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the forward log1p stretch."""
        return super().transform(x, params)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return LogP1Transform(log_fn=self.fn)


class MinMaxScale(Transform):
    """Affine rescale from `[vmin, vmax]` into `feature_range`."""

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ):
        """Constructor.

        Args:
            vmin: Input minimum; `None` uses each tensor's own `min()`.
            vmax: Input maximum; `None` uses each tensor's own `max()`.
            feature_range: Target `(low, high)` range.
        """
        super().__init__()
        self.vmin = None if vmin is None else _as_tensor(vmin)
        self.vmax = None if vmax is None else _as_tensor(vmax)
        self.lo, self.hi = feature_range

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Rescale the input into `feature_range`."""
        vmin = x.min() if self.vmin is None else self.vmin
        vmax = x.max() if self.vmax is None else self.vmax
        x = (x - vmin) / (vmax - vmin)
        return x * (self.hi - self.lo) + self.lo

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Invert the rescale (requires fixed `vmin`/`vmax`)."""
        if self.vmin is None or self.vmax is None:
            raise RuntimeError("MinMaxScale.revert requires fixed vmin/vmax.")
        x = (x - self.lo) / (self.hi - self.lo)
        return x * (self.vmax - self.vmin) + self.vmin

    def get_inverse(self) -> Transform:
        """Return the inverse rescale (requires fixed `vmin`/`vmax`)."""
        if self.vmin is None or self.vmax is None:
            raise RuntimeError("MinMaxScale.get_inverse requires fixed vmin/vmax.")
        return InvMinMaxScale(
            vmin=self.vmin, vmax=self.vmax, feature_range=(self.lo, self.hi)
        )


class InvMinMaxScale(MinMaxScale):
    """Inverse of `MinMaxScale` (requires fixed `vmin`/`vmax`)."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the inverse rescale."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Apply the forward rescale."""
        return super().transform(x, params)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return MinMaxScale(
            vmin=self.vmin, vmax=self.vmax, feature_range=(self.lo, self.hi)
        )


class Clamp(Transform):
    """Clamp values to `[min, max]` (lossy, no inverse)."""

    _transformed_types = (torch.Tensor,)

    def __init__(self, min: float | None = None, max: float | None = None):
        """Constructor.

        Args:
            min: Lower bound (`None` leaves the lower side unbounded).
            max: Upper bound (`None` leaves the upper side unbounded).
        """
        super().__init__()
        self.min = min
        self.max = max

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Clamp the input."""
        return torch.clamp(x, min=self.min, max=self.max)


class HounsfieldScale(Transform):
    """Rescale raw CT stored values to Hounsfield units: `slope * x + intercept`."""

    _transformed_types = (torch.Tensor,)

    def __init__(self, slope: float = 1.0, intercept: float = -1024.0):
        """Constructor.

        Args:
            slope: DICOM `RescaleSlope` (tag 0028,1053).
            intercept: DICOM `RescaleIntercept` (tag 0028,1052).
        """
        super().__init__()
        self.slope = _as_tensor(slope)
        self.intercept = _as_tensor(intercept)
        self.affine = Affine(a=1.0 / self.slope, b=-self.intercept / self.slope)

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Map stored values to Hounsfield units."""
        return self.affine.transform(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Map Hounsfield units back to stored values."""
        return self.affine.revert(x, params)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return InvHounsfieldScale(slope=self.slope, intercept=self.intercept)


class InvHounsfieldScale(HounsfieldScale):
    """Inverse of `HounsfieldScale` (Hounsfield units -> stored values)."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Map Hounsfield units to stored values."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Map stored values to Hounsfield units."""
        return super().transform(x, params)

    def get_inverse(self) -> Transform:
        """Return the inverse transform."""
        return HounsfieldScale(slope=self.slope, intercept=self.intercept)


class HounsfieldClamp(Clamp):
    """Clamp Hounsfield units to a window (lossy, no inverse)."""

    def __init__(self, center: float, width: float):
        """Constructor.

        Args:
            center: Window center (level) in HU, e.g. 40 for soft tissue.
            width: Window width in HU, e.g. 400 for soft tissue.
        """
        super().__init__(min=center - width / 2, max=center + width / 2)
        self.center = center
        self.width = width


def _lstsq_line(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit `y ≈ slope * x + intercept`; returns `(slope, intercept)`."""
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / denom
    return slope, ym - slope * xm


def _grow(mask: torch.Tensor, ngrow: int) -> torch.Tensor:
    """Dilate a 1-D boolean mask by a window of `ngrow` (max-filter)."""
    if ngrow <= 1:
        return mask
    m = mask.float().view(1, 1, -1)
    pooled = F.max_pool1d(m, kernel_size=ngrow, stride=1, padding=ngrow // 2)
    return pooled.view(-1)[: mask.numel()] > 0


class ZScaleInterval:
    """Compute IRAF zscale display limits `(z1, z2)` from a tensor.

    Samples the data, fits a line to the sorted samples with iterative k-sigma
    rejection, and derives cut limits about the median (see IRAF/`astropy`
    `ZScaleInterval`).
    """

    def __init__(
        self,
        nsamples: int = 1000,
        contrast: float = 0.25,
        max_reject: float = 0.5,
        min_npixels: int = 5,
        krej: float = 2.5,
        max_iterations: int = 5,
    ):
        """Constructor.

        Args:
            nsamples: Number of pixels to sample from the input.
            contrast: Scales the fitted slope; smaller means higher contrast.
            max_reject: Minimum fraction of samples that must survive
                rejection for the fit to be used (IRAF convention;
                despite the name).
            min_npixels: Minimum number of surviving samples for a valid fit.
            krej: k-sigma rejection threshold on the fit residuals.
            max_iterations: Maximum number of rejection iterations.
        """
        self.nsamples = nsamples
        self.contrast = contrast
        self.max_reject = max_reject
        self.min_npixels = min_npixels
        self.krej = krej
        self.max_iterations = max_iterations

    def get_limits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the `(z1, z2)` cut limits for `x`."""
        values = x.flatten()
        values = values[torch.isfinite(values)]
        stride = int(max(1.0, values.numel() / self.nsamples))
        samples = torch.sort(values[::stride][: self.nsamples])[0].float()
        npix = samples.numel()
        vmin, vmax = samples[0], samples[-1]
        if npix < self.min_npixels:
            return vmin, vmax

        minpix = max(self.min_npixels, int(npix * self.max_reject))
        ngrow = max(1, int(npix * 0.01))
        idx = torch.arange(npix, dtype=samples.dtype, device=samples.device)
        badpix = torch.zeros(npix, dtype=torch.bool, device=samples.device)
        slope = torch.zeros((), dtype=samples.dtype, device=samples.device)
        ngoodpix, last_ngoodpix = npix, npix + 1
        for _ in range(self.max_iterations):
            if ngoodpix >= last_ngoodpix or ngoodpix < minpix:
                break
            good = ~badpix
            slope, intercept = _lstsq_line(idx[good], samples[good])
            resid = samples - (slope * idx + intercept)
            # population std (ddof=0), as in IRAF/`astropy`; torch would
            # otherwise default to the sample std (correction=1)
            threshold = self.krej * resid[good].std(correction=0)
            badpix = _grow(badpix | (resid.abs() > threshold), ngrow)
            last_ngoodpix, ngoodpix = ngoodpix, int((~badpix).sum())

        if ngoodpix < minpix:
            return vmin, vmax
        if self.contrast > 0:
            slope = slope / self.contrast
        center = (npix - 1) // 2
        median = samples.median()
        z1 = torch.maximum(vmin, median - center * slope)
        z2 = torch.minimum(vmax, median + (npix - 1 - center) * slope)
        return z1, z2


class ZScale(Transform):
    """Adaptive IRAF zscale normalization: clip to zscale limits, then rescale."""

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        interval: ZScaleInterval | None = None,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ):
        """Constructor.

        Args:
            interval: The `ZScaleInterval` computing `(z1, z2)` (default: one with
                IRAF defaults).
            feature_range: Target `(low, high)` range after clipping.
        """
        super().__init__()
        self.interval = interval or ZScaleInterval()
        self.feature_range = feature_range

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Compute the shared `(z1, z2)` limits from the first tensor."""
        x = next(i for i in flat_inputs if isinstance(i, torch.Tensor))
        z1, z2 = self.interval.get_limits(x)
        return {"z1": z1, "z2": z2}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        """Clip to `[z1, z2]` and rescale into `feature_range`."""
        z1, z2 = params["z1"], params["z2"]
        x = Clamp(z1, z2).transform(x)
        return MinMaxScale(z1, z2, self.feature_range).transform(x)

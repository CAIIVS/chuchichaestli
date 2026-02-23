# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Procedurally generated datasets.

Each dataset is a `GenerativeDataset` subclass. Subclasses implement
`GenerativeDataset.generate` which returns `(X, y)` tensors; the base
class handles SharedArray allocation, optional SafeTensors persistence, and the
standard PyTorch Dataset protocol.

Generators provided:
- `MoonsDataset`
- `SpiralsDataset`
- `CheckerboardDataset`
- `RingsDataset`
- `ConcentricSpheresDataset`
- `GaussiansDataset`
- `SwissRollDataset`
"""

from pathlib import Path
import torch
from torch.distributions import Normal
from abc import ABC, abstractmethod
import warnings
from collections.abc import Callable
from chuchichaestli.data.base import CachingDataset, DataReturnTypes
from chuchichaestli.data.cache import nbytes


__all__ = [
    "ProceduralDataset",
    "HalfMoonsDataset",
    "SpiralsDataset",
    "CheckerboardDataset",
    "RingsDataset",
    "ConcentricSpheresDataset",
    "GaussiansDataset",
    "SwissRollDataset",
    "generate_procedural_dataset",
]


class ProceduralDataset(CachingDataset, ABC):
    """Base class for procedurally generated labelled datasets.

    Subclasses implement `generate`, which returns `(X, y)` tensors.
    All data is stored in shared memory cache and can be safely shared
    across PyTorch DataLoader worker processes.
    When path points to a `.safetensors` file the data is loaded from disk;
    otherwise `generate` is called at construction time.  Call `save` afterwards
    to persist a freshly generated dataset.
    """

    FILE_EXTENSIONS: list[str] = [".safetensors"]

    def __init__(
        self,
        dim: int,
        n_samples: int,
        path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        cache: int | float | str | bool | nbytes | None = None,
        seed: int = 42,
        **kwargs,
    ):
        """Constructor.

        Args:
            dim: Feature dimensionality.
            n_samples: Total number of samples in the dataset.
            path: Optional path to a `.safetensors` file.  When the file exists
                its contents are loaded instead of calling `generate`.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for `CachingDataset`.
        """
        self._req_samples = n_samples
        self.dim = dim
        self.seed = seed

        if cache is None:
            elem_bytes = torch.tensor(0, dtype=dtype).element_size()
            cache = n_samples * (dim + 1) * elem_bytes

        super().__init__(
            path=path, dtype=dtype, return_as=return_as, cache=cache, **kwargs
        )

        if not self.has_files:
            if path is not None and not Path(path).is_dir():
                warnings.warn(
                    f"{type(self).__name__}: path {str(path)!r} does not exist; "
                    "generating data. Call save() for persistance.",
                    UserWarning,
                    stacklevel=3,
                )
            self._init_from_tensors(*self.generate())

    @staticmethod
    def _partition(n_samples: int, n_classes: int) -> list[int]:
        """Distribute `n_samples` as evenly as possible across `n_classes`.

        The remainder is spread one-by-one over the first classes, so the
        returned counts always sum to exactly *n_samples*.
        """
        base, remainder = divmod(n_samples, n_classes)
        return [base + (1 if i < remainder else 0) for i in range(n_classes)]

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return super().n_samples if self._mmap else self._req_samples

    def _init_from_tensors(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Populate `_mmap` from `(X, y)` tensors and (re)init the cache.

        Args:
            X: Feature tensor of shape `(n, dim)`.
            y: Label tensor of shape `(n,)`.
        """
        data = torch.cat([X.to(self.dtype), y.to(self.dtype).unsqueeze(1)], dim=1)
        self._mmap = [data]
        self._build_index()
        self.init_cache()

    def load(self, **kwargs) -> None:
        """Load data from the first resolved `.safetensors` file."""
        from safetensors.torch import load_file

        if not self.files:
            return
        path = self.files[0]
        if self.n_files > 1:
            warnings.warn(
                f"{type(self).__name__}: {len(self.files)} files found; "
                f"loading {path.name} only.",
                UserWarning,
                stacklevel=2,
            )

        tensors = load_file(str(path))
        X = tensors["X"].to(self.dtype)
        y = tensors["y"].to(self.dtype)
        # Check number of samples vs.
        file_n = X.shape[0]
        if file_n != self._req_samples:
            warnings.warn(
                f"{type(self).__name__}: n_samples={self._req_samples} but "
                f"{path.name} contains {file_n} samples; using {file_n}.",
                UserWarning,
                stacklevel=2,
            )
            self._req_samples = file_n

        data = torch.cat([X, y.unsqueeze(1)], dim=1)
        self._mmap = [data]
        self._build_index()

    def close(self, **kwargs):
        """Release in-memory data and purge the slot cache."""
        self._mmap = []
        self._file_offsets = []
        super().close()

    @abstractmethod
    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the dataset from scratch.

        Returns:
            `(X, y)` where X has shape `(n_samples, dim)` and y `(n_samples,)`.
        """
        ...

    def regenerate(self, seed: int | None = None) -> None:
        """Regenerate data in place, optionally with a new seed.

        Replaces ``self._mmap`` and reinitialises the slot cache.  Useful for
        data-augmentation loops or hyperparameter sweeps.

        Args:
            seed: New random seed.  ``None`` reuses ``self.seed``, reproducing
                the original data exactly.
        """
        if seed is not None:
            self.seed = seed
        self._init_from_tensors(*self.generate())

    def save(self, path: str | Path) -> Path:
        """Persist the dataset to a `.safetensors` file.

        Args:
            path: Destination file path.
        """
        if not self._mmap:
            raise RuntimeError(f"{type(self).__name__}: no data to save.")

        from safetensors.torch import save_file  # lazy import

        path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._mmap[0]
        save_file(
            {
                "X": data[:, : self.dim].clone(),
                "y": data[:, self.dim].clone(),
            },
            str(path),
        )
        return path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return item as `(features, label)` for index."""
        row = super().__getitem__(idx)
        return row[: self.dim], row[self.dim]

    def __repr__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}(n_samples={self.n_samples}, dim={self.dim})"


class HalfMoonsDataset(ProceduralDataset):
    """Two interleaving half-moon point clouds."""

    def __init__(
        self,
        n_samples: int = 1000,
        noise: float = 0.05,
        path: str | Path | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            n_samples: Total number of samples (split evenly between moons).
            noise: Standard deviation of Gaussian noise added to each sample.
            path: Optional `.safetensors` file to load from / save to.
            dim: Feature dimensionality.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        kwargs.setdefault("dim", 2)
        self.noise = noise
        super().__init__(
            n_samples=n_samples,
            path=path,
            **kwargs,
        )

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate half-moon samples."""
        torch.manual_seed(self.seed)
        n_out, n_in = self._partition(self.n_samples, 2)
        # Outer moon: upper half-circle.
        t_out = torch.linspace(0, torch.pi, n_out)
        X_out = torch.stack([torch.cos(t_out), torch.sin(t_out)], dim=1)
        # Inner moon: lower half-circle, offset so the moons interleave.
        t_in = torch.linspace(0, torch.pi, n_in)
        X_in = torch.stack([1.0 - torch.cos(t_in), 0.5 - torch.sin(t_in)], dim=1)
        # Build data tensors
        X = torch.cat([X_out, X_in])
        y = torch.cat([torch.zeros(n_out), torch.ones(n_in)])
        # Add random jitter on 2D plane
        if self.noise > 0:
            X = X + torch.randn_like(X) * self.noise
        # Embed into higher-diemnsional ambient space
        if self.dim > 2:
            extra = torch.randn(self.n_samples, self.dim - 2) * self.noise
            X = torch.cat([X, extra], dim=1)
        return X, y


class SpiralsDataset(ProceduralDataset):
    """Two interleaved Archimedean spiral point clouds."""

    def __init__(
        self,
        n_samples: int = 1000,
        noise: float = 0.05,
        path: str | Path | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            n_samples: Total number of samples (split evenly between spiral arms).
            noise: Standard deviation of Gaussian noise added to each sample.
            path: Optional `.safetensors` file.
            dim: Feature dimensionality.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        kwargs.setdefault("dim", 2)
        self.noise = noise
        super().__init__(
            n_samples=n_samples,
            path=path,
            **kwargs,
        )

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate two-spiral samples."""
        torch.manual_seed(self.seed)
        n0, n1 = self._partition(self.n_samples, 2)
        # Angular parameter (sqrt-sampling gives uniform density along the arc)
        t0 = torch.sqrt(torch.rand(n0)) * 780.0 * (2.0 * torch.pi / 360.0)
        t1 = torch.sqrt(torch.rand(n1)) * 780.0 * (2.0 * torch.pi / 360.0)
        # First arm.
        jitter0 = torch.randn(n0, 2) * self.noise
        arm0 = torch.stack([-torch.cos(t0) * t0, torch.sin(t0) * t0], dim=1) + jitter0
        # Second arm, 180 degrees rotation with independent noise.
        jitter1 = torch.randn(n1, 2) * self.noise
        arm1 = -torch.stack([-torch.cos(t1) * t1, torch.sin(t1) * t1], dim=1) + jitter1
        # Build data tensors
        X = torch.cat([arm0, arm1])
        y = torch.cat([torch.zeros(n0), torch.ones(n1)])
        # Embed into higher-dimensional ambient space if needed
        if self.dim > 2:
            extra = torch.randn(self.n_samples, self.dim - 2) * self.noise
            X = torch.cat([X, extra], dim=1)
        return X, y


class CheckerboardDataset(ProceduralDataset):
    """Axis-aligned checkerboard pattern dataset."""

    def __init__(
        self,
        n_samples: int = 1000,
        n_tiles: int = 4,
        extent: float = 2.0,
        noise: float = 0.0,
        path: str | Path | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            n_samples: Total number of samples.
            n_tiles: Number of tiles along each axis.
            extent: Half-width of the sampling region.
            noise: Standard deviation of Gaussian noise added to each sample.
            path: Optional `.safetensors` file.
            dim: Feature dimensionality.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        kwargs.setdefault("dim", 2)
        self.n_tiles = n_tiles
        self.extent = extent
        self.noise = noise
        super().__init__(
            n_samples=n_samples,
            path=path,
            **kwargs,
        )

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate checkerboard samples."""
        torch.manual_seed(self.seed)
        e = self.extent
        X = torch.empty(self.n_samples, self.dim).uniform_(-e, e)
        cell_size = 2.0 * e / self.n_tiles
        # Sum tile-index parities across all dimensions
        indices = torch.floor((X + e) / cell_size).long()
        y = (indices.sum(dim=1) % 2).float()
        if self.noise > 0:
            X = X + torch.randn_like(X) * self.noise
        return X, y


class RingsDataset(ProceduralDataset):
    """Concentric rings, each ring a separate class."""

    def __init__(
        self,
        n_samples: int = 1000,
        n_rings: int = 3,
        inner_radius: float = 0.5,
        ring_spacing: float = 0.8,
        width: float = 0.1,
        noise: float = 0.0,
        path: str | Path | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            n_samples: Total number of samples.
            n_rings: Number of concentric rings/classes.
            inner_radius: Radius of the innermost ring.
            ring_spacing: Radial gap between consecutive ring centres.
            width: Radial half-width (thickness) of each ring.
            noise: Standard deviation of Gaussian noise added to each sample.
            path: Optional `.safetensors` file.
            dim: Feature dimensionality.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        kwargs.setdefault("dim", 2)
        self.n_rings = n_rings
        self.inner_radius = inner_radius
        self.ring_spacing = ring_spacing
        self.width = width
        self.noise = noise
        super().__init__(
            n_samples=n_samples,
            path=path,
            **kwargs,
        )

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate concentric-ring samples."""
        torch.manual_seed(self.seed)
        n = self.n_samples // self.n_rings
        counts = self._partition(self.n_samples, self.n_rings)
        X_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        # Build each ring with spacing inbetween
        for k, n in enumerate(counts):
            radius = self.inner_radius + k * self.ring_spacing
            theta = torch.empty(n).uniform_(0.0, 2.0 * torch.pi)
            r = radius + torch.empty(n).uniform_(-self.width, self.width)
            pts = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            if self.noise > 0:
                pts = pts + torch.randn_like(pts) * self.noise
            if self.dim > 2:
                extra = torch.randn(n, self.dim - 2) * self.noise
                pts = torch.cat([pts, extra], dim=1)
            X_parts.append(pts)
            y_parts.append(torch.full((n,), k, dtype=self.dtype))
        return torch.cat(X_parts), torch.cat(y_parts)


class ConcentricSpheresDataset(ProceduralDataset):
    """Inner and outer concentric (hyper-)spheres."""

    def __init__(
        self,
        dim: int = 3,
        n_samples: int = 1000,
        inner_radius: float = 0.5,
        outer_radius: float = 1.0,
        noise: float = 0.05,
        path: str | Path | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            dim: Ambient dimensionality.
            n_samples: Total number of samples.
            inner_radius: Radius of the inner sphere.
            outer_radius: Radius of the outer sphere.
            noise: Standard deviation of Gaussian noise added to each sample.
            path: Optional `.safetensors` file.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        self.noise = noise
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        super().__init__(
            dim=dim,
            n_samples=n_samples,
            path=path,
            **kwargs,
        )

    @staticmethod
    def _randnsphere(dim: int, n: int, radius: float = 1.0) -> torch.Tensor:
        """Sample n points uniformly on the surface of a `dim`-sphere.

        Args:
            dim: Dimensionality of the ambient space.
            n: Number of points to sample.
            radius: Radius of the sphere.
        """
        v = torch.randn(n, dim)
        return v * (radius / v.norm(dim=1, keepdim=True))

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate concentric-sphere samples."""
        torch.manual_seed(self.seed)
        n_inner, n_outer = self._partition(self.n_samples, 2)
        X_inner = self._randnsphere(self.dim, n_inner, self.inner_radius)
        X_outer = self._randnsphere(self.dim, n_outer, self.outer_radius)
        if self.noise > 0:
            X_inner = X_inner + torch.randn_like(X_inner) * self.noise
            X_outer = X_outer + torch.randn_like(X_outer) * self.noise
        X = torch.cat([X_inner, X_outer])
        y = torch.cat([torch.zeros(n_inner), torch.ones(n_outer)])
        return X, y


class GaussiansDataset(ProceduralDataset):
    """Isotropic Gaussians arranged on a 2D ring."""

    def __init__(
        self,
        dim: int = 2,
        n_samples: int = 1000,
        n_gaussians: int = 6,
        radius: float = 1.0,
        std: float = 0.1,
        noise: float = 0.01,
        path: str | Path | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            dim: Ambient dimensionality.
            n_samples: Total number of samples.
            n_gaussians: Number of Gaussian blobs/classes.
            radius: Radius of the ring on which the Gaussian centres lie.
            std: Standard deviation of each Gaussian.
            noise: Standard deviation of Gaussian jitter added to each sample.
            path: Optional `.safetensors` file.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        self.n_gaussians = n_gaussians
        self.radius = radius
        self.std = std
        self.noise = noise
        super().__init__(
            dim=dim,
            n_samples=n_samples,
            path=path,
            **kwargs,
        )

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate ring-of-Gaussians samples."""
        torch.manual_seed(self.seed)
        X_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        counts = self._partition(self.n_samples, self.n_gaussians)
        for k, n in enumerate(counts):
            # Coordinates of the Gaussian
            angle = torch.tensor(2.0 * torch.pi * k / self.n_gaussians)
            centre = torch.zeros(self.dim)
            centre[0] = self.radius * torch.cos(angle)
            centre[1] = self.radius * torch.sin(angle)
            # Gaussian data
            pts = Normal(centre, torch.full((self.dim,), self.std)).sample((n,))
            # Additional jitter
            if self.noise > 0:
                pts = pts + torch.randn_like(pts) * self.noise
            X_parts.append(pts)
            y_parts.append(torch.full((n,), k, dtype=self.dtype))
        return torch.cat(X_parts), torch.cat(y_parts)


class SwissRollDataset(ProceduralDataset):
    """Spiral manifold embedded in 3D; winding parameter as continuous label."""

    def __init__(
        self,
        n_samples: int = 1000,
        height: float = 1.0,
        t_min: float = 1.5 * torch.pi,
        t_max: float = 4.5 * torch.pi,
        noise: float = 0.05,
        path: str | Path | None = None,
        **kwargs
    ):
        """Constructor.

        Args:
            n_samples: Total number of samples.
            height: Extent of the roll along the y-axis.
            t_min: Minimum winding angle (in radians).
            t_max: Maximum winding angle (in radians).
            noise: Standard deviation of Gaussian noise added to each sample.
            path: Optional `.safetensors` file.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            seed: Random seed for reproducible generation.
            kwargs: Additional keywords for parent class constructor.
        """
        kwargs.setdefault("dim", 3)
        self.height = height
        self.t_min = t_min
        self.t_max = t_max
        self.noise = noise
        super().__init__(
            n_samples=n_samples,
            path=path,
            **kwargs
        )

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Swiss-roll samples."""
        torch.manual_seed(self.seed)
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(self.n_samples)
        h = self.height * (torch.rand(self.n_samples) - 0.5)
        # Build spiral
        X = torch.stack([t * torch.cos(t), h, t * torch.sin(t)], dim=1)
        # Additional jitter
        if self.noise > 0:
            X = X + torch.randn_like(X) * self.noise
        # Embed into higher-dimensional ambient space if needed
        if self.dim > 3:
            extra = torch.randn(self.n_samples, self.dim - 3) * self.noise
            X = torch.cat([X, extra], dim=1)
        # Normalised winding angle as a continuous label.
        y = (t - self.t_min) / (self.t_max - self.t_min)
        return X, y


def generate_procedural_dataset(
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    dim: int,
    n_samples: int,
    path: str | Path | None = None,
    **kwargs,
) -> ProceduralDataset:
    """Create a `ProceduralDataset` instance from any callable.

    Wraps function in a dynamically created `ProceduralDataset` subclass as the
    `ProceduralDataset.generate` method. The callable receives the dataset
    instance as context so it has access to `n_samples`, `dim`, `seed`, and any
    extra attributes in the instance context.

    Args:
        fn: Callable with signature `fn(dataset) -> (X, y)`;
            `X` must have shape `(n_samples, dim)` and `y` shape `(n_samples,)`.
        dim: Feature dimensionality.
        n_samples: Total number of samples.
        path: Optional `.safetensors` file to load from/save to.
        **kwargs: Extra keyword arguments stored as instance attributes.
    """
    # Unknown kwargs are set as attributes before generate() is called.
    _base_kwargs = {"dtype", "return_as", "cache", "seed"}
    extra_attrs = {k: v for k, v in kwargs.items() if k not in _base_kwargs}
    base_kwargs = {k: v for k, v in kwargs.items() if k in _base_kwargs}

    class _CustomProcDataset(ProceduralDataset):
        def __init__(self):
            # Extra keywords as attributes before `generate` is called
            for attr, val in extra_attrs.items():
                setattr(self, attr, val)
            super().__init__(
                dim=dim,
                n_samples=n_samples,
                path=path,
                **base_kwargs,
            )

        def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
            return fn(self)

    _CustomProcDataset.__name__ = getattr(fn, "__name__", "CustomProcDataset")
    _CustomProcDataset.__qualname__ = _CustomProcDataset.__name__
    return _CustomProcDataset()

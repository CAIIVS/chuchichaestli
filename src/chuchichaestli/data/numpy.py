# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch dataset classes for numpy file reading."""

from pathlib import Path
import fnmatch
import numpy as np
import numpy.lib.format as nf
import torch
import threading
import warnings
from io import RawIOBase
from collections.abc import Sequence
from chuchichaestli.data.cache import nbytes
from chuchichaestli.data.base import CachingDataset, DataReturnTypes
from chuchichaestli.data.zip import ZipDataset


__all__ = ["NumpyDataset", "ZipNumpyDataset"]


class NpyArrayView:
    """Proxy for a single .npy file.

    Supports integer indexing (`view[i]`) and exposes `.shape` and `.dtype`
    so that it behaves like a memory-mapped `numpy.ndarray`.
    """

    def __init__(self, file_path: Path):
        self._path = file_path
        self._local = threading.local()
        # Open briefly to peek at header metadata.
        with open(file_path, "rb") as f:
            version = nf.read_magic(f)
            if version == (1, 0):
                shape, fortran_order, dtype = nf.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran_order, dtype = nf.read_array_header_2_0(f)
            else:
                raise ValueError(
                    f"{file_path}: unsupported .npy format version {version}"
                )
            self._data_offset: int = f.tell()
        self._fortran_order: bool = fortran_order
        # if fortran_order:
        #     raise ValueError(f"{file_path}: Fortran-order arrays are not supported")
        self.shape: tuple[int, ...] = shape
        self.dtype: np.dtype = dtype
        self._sample_shape: tuple[int, ...] = shape[1:]
        self._item_bytes: int = int(np.prod(shape[1:])) * dtype.itemsize
        self._sample_elems: int = int(np.prod(shape[1:]))

    def __len__(self) -> int:
        return self.shape[0]

    def _get_fd(self) -> RawIOBase:
        """Return the cached file descriptor for the current thread.

        Opens a new fd on first access, or after `flush()` has closed it.
        """
        fd = getattr(self._local, "fd", None)
        if fd is None or fd.closed:
            self._local.fd = open(self._path, "rb")
        return self._local.fd

    def flush(self) -> None:
        """Close the thread-local file descriptor.

        Called by `FileDataset._read_item` after every sample access.
        The fd will be transparently re-opened on the next `__getitem__` call.
        """
        fd = getattr(self._local, "fd", None)
        if fd is not None and not fd.closed:
            fd.close()

    def close(self) -> None:
        """Alias for `flush`; called explicitly during dataset teardown."""
        self.flush()

    def _read_fortran_sample(self, f, idx: int) -> np.ndarray:
        """Read a single Fortran-order sample without extra file handles."""
        N = self.shape[0]
        M = self._sample_elems
        f.seek(self._data_offset)
        buf = f.read(N * M * self.dtype.itemsize)
        flat = np.frombuffer(buf, dtype=self.dtype)
        # Extract elements at flat positions idx, idx+N, idx+2N, ...
        indices = idx + np.arange(M, dtype=np.intp) * N
        return flat[indices].reshape(self._sample_shape).copy()

    def __getitem__(self, idx) -> np.ndarray:
        """Open the file, copy the requested slice, close immediately."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            indices = range(start, stop, step)
            n = len(indices)
            # Fortran-order contiguous slice
            if self._fortran_order:
                if n == 0:
                    return np.empty((0, *self._sample_shape), dtype=self.dtype)
                return np.stack([self[i] for i in indices])
            # C-order contiguous slice
            f = self._get_fd()
            if step == 1:
                f.seek(self._data_offset + start * self._item_bytes)
                buf = f.read(n * self._item_bytes)
                return np.frombuffer(buf, dtype=self.dtype).reshape((n, *self._sample_shape)).copy()
            # Non-contiguous slice: read sample by sample
            return np.stack([self[i] for i in indices])
        f = self._get_fd()
        if self._fortran_order:
            return self._read_fortran_sample(f, idx)
        f.seek(self._data_offset + idx * self._item_bytes)
        buf = f.read(self._item_bytes)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self._sample_shape).copy()


class NumpyDataset(CachingDataset):
    """Dataset for loading numpy files (.npy / .npz) with (sto)caching features.

    Features:
    - lazy .npy file access via `NpyArrayView`
    - .npz archive support with key-pattern selection and array concatenation
    - shared memory caching via CachingDataset
    - optional attribute/metadata loading
      - for .npy files: sidecar `<stem>.attrs.npy` file
      - for .npz files: matching key pattern inside the archive
    - multi-file support with automatic index fusing
    """

    FILE_EXTENSIONS = [".npy", ".npz"]

    def __init__(
        self,
        path: str | Path | Sequence[str] | Sequence[Path],
        keys: str | Sequence[str] = "*",
        attrs_keys: str | Sequence[str] | None = None,
        attrs_suffix: str = ".attrs",
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        cache: int | float | str | bool | nbytes | None = "4G",
        attrs_cache: int | float | str | bool | nbytes | None = None,
        preload: bool = False,
        new_axis: bool = False,
        **kwargs,
    ):
        """Constructor.

        Args:
            path: Path to file(s) or data directory. May contain wildcards (`*` and `**`).
            keys: Key pattern(s) for selecting arrays.
                - For `.npy` files, this argument is ignored.
                - For `.npz` files, use `"*"` for all keys, `"path/*"` for subgroups, etc.
            attrs_keys: Key pattern(s) for attribute/metadata arrays.
                - For `.npy` files, pattern matched against sidecar file
                  names of the form `<stem>.attrs.npy` in the same directory
                  (use "*" to pick up any sidecar).
                - For `.npz` files, matched against archive keys, exactly
                  like `keys`. Matched keys are excluded from data keys.
                If `None`, no attributes are loaded.
            attrs_suffix: Suffix for .npy files that contain metadata, e.g.
                `'.attrs'` becomes `'<stem>.attrs.npy'`.
            dtype: Data tensor type; default: `torch.float32`.
            return_as: Return type; one of `['tuple', 'dict']` or a custom dict mapping.
                - `'tuple'`: Returns tuple of samples (default).
                - `'dict'`: Returns dict with key `'data'` (and
                  `'attrs'` when attributes are present).
                - `dict`: Custom mapping, e.g. `{'input': 0, 'target': 1}`.
            cache: Cache size for data items (e.g. `"4G"`, `4.0`, or bytes).
            attrs_cache: Cache size for attributes/metadata.
            preload: Preload and cache the dataset.
            new_axis: If `True`, each file is one sample; see `FileDataset`
                for full documentation.
            kwargs: Reserved for forward-compatibility.
        """
        # Key patterns
        self.key_patterns: tuple[str, ...] = (
            (keys,) if isinstance(keys, str) else tuple(keys)
        )
        self.attrs_patterns: tuple[str, ...] | None = None
        if attrs_keys is not None:
            self.attrs_patterns = (
                (attrs_keys,) if isinstance(attrs_keys, str) else tuple(attrs_keys)
            )
        self.attrs_suffix = attrs_suffix
        # Initialize CachingDataset
        super().__init__(
            path=path,
            dtype=dtype,
            return_as=return_as,
            cache=cache,
            attrs_cache=attrs_cache,
            preload=preload,
            copy_on_write=False,
            has_attrs=attrs_keys is not None,
            new_axis=new_axis,
        )

    def load(self, **kwargs):
        """Load NumPy files and map arrays.

        Open all files, select arrays matching key patterns, concatenate
        multiple matching arrays into a single contiguous array, and build
        the sequential index across files.
        """
        # Issue warning if keys have been passed, but files only include .npy
        npy_files = [f for f in self.files if f.suffix == ".npy"]
        npz_files = [f for f in self.files if f.suffix == ".npz"]
        
        if npy_files and not npz_files and self.key_patterns != ("*",):
            warnings.warn(
                f"key_patterns {self.key_patterns!r} are ignored for .npy files, "
                "which each contain exactly one array.",
                UserWarning,
                stacklevel=2,
            )
        # Load memory maps depending on file type
        for file_path in self.files:
            if file_path.suffix == ".npy":
                self._load_npy(file_path)
            else:
                self._load_npz(file_path)
        self._build_index()

    def _load_npy(self, file_path: Path):
        """Load a single .npy file.

        The array is memory-mapped. Attribute sidecar files (`<stem>.attrs.npy`)
        are loaded if `attrs_patterns` is set.
        """
        data = NpyArrayView(file_path)
        self._mmap.append(data)

        # Look for sidecar data matching attrs_patterns
        if self.attrs_patterns is not None:
            sidecar = file_path.parent / f"{file_path.stem}{self.attrs_suffix}.npy"
            if sidecar.exists():
                metadata = np.load(sidecar, allow_pickle=True)
                self._mmap_attrs.append(metadata)
            else:
                self._mmap_attrs.append(None)  # placeholder to keep index aligned

    def _load_npz(self, file_path: Path):
        """Load arrays from a .npz archive matching key patterns.

        Selected data arrays are concatenated along axis 0 when more than one
        key matches. Attrs arrays are loaded separately.
        """
        archive = np.load(file_path, allow_pickle=False)
        all_keys: list[str] = list(archive.files)
        # Collect attribute keys first so they can be excluded from data keys
        attrs_keys: list[str] = []
        if self.attrs_patterns is not None:
            attrs_keys = self._match_keys(all_keys, self.attrs_patterns)

        data_keys = self._match_keys(
            [k for k in all_keys if k not in attrs_keys], self.key_patterns
        )
        # Build data mmap (concatenate if multiple keys match)
        if data_keys:
            arrays = [archive[k] for k in data_keys]
            data = (
                arrays[0].copy()
                if len(arrays) == 1
                else self._concat_arrays(arrays, file_path)
            )
            self._mmap.append(data)
        # if no data_keys, nothing is appended; _build_index skips empty mmaps
        if self.attrs_patterns is not None:
            if attrs_keys:
                attrs_arrays = [archive[k] for k in attrs_keys]
                attrs = (
                    attrs_arrays[0].copy()
                    if len(attrs_arrays) == 1
                    else self._concat_arrays(attrs_arrays, file_path)
                )
                self._mmap_attrs.append(attrs)
            else:
                self._mmap_attrs.append(None)
        archive.close()

    @staticmethod
    def _match_keys(keys: list[str], patterns: Sequence[str]) -> list[str]:
        """Return keys matching at least one fnmatch pattern (ordered, deduplicated).

        Args:
            keys: Candidate key names.
            patterns: Matching patterns, compatible with wildcards (* and **).
        """
        matched: list[str] = []
        for pattern in patterns:
            for key in keys:
                if fnmatch.fnmatch(key, pattern) and key not in matched:
                    matched.append(key)
        return matched

    @staticmethod
    def _concat_arrays(arrays: list[np.ndarray], source: Path) -> np.ndarray:
        """Concatenate arrays along axis 0 with shape/dtype validation.

        Args:
            arrays: NumPy arrays to concatenate.
            source: Source file path (used in error messages only).
        """
        ref_shape = arrays[0].shape[1:]
        ref_dtype = arrays[0].dtype
        for i, arr in enumerate(arrays[1:], start=1):
            if arr.shape[1:] != ref_shape:
                raise ValueError(
                    f"Array shapes are incompatible for concatenation in '{source}': "
                    f"array 0 has shape {arrays[0].shape}, "
                    f"array {i} has shape {arr.shape}"
                )
            if arr.dtype != ref_dtype:
                raise ValueError(
                    f"Array dtypes are incompatible for concatenation in '{source}': "
                    f"array 0 has dtype {ref_dtype}, "
                    f"array {i} has dtype {arr.dtype}"
                )
        return np.concatenate(arrays, axis=0)

    def close(self):
        """Close memory maps and clean up resources."""
        for entry in self._mmap:
            if isinstance(entry, NpyArrayView):
                entry.close()
            elif isinstance(entry, np.memmap):
                try:
                    entry._mmap.close()
                except Exception:
                    pass
        self._mmap.clear()
        self._mmap_attrs.clear()
        self._file_offsets.clear()
        # Purge shared-memory caches
        super().close()

    @property
    def n_datasets(self) -> int:
        """Total number of array groups (keys) across all files."""
        return len(self._mmap)

    def info(self, print_: bool = True) -> str:
        """Print dataset information.

        Args:
            print_: Whether to print to stdout.
        """
        summary = str(self) + "\n"
        summary += "-" * 50 + "\n"
        summary += f"Files:              {self.n_files}\n"
        summary += f"Datasets:           {self.n_datasets}\n"
        summary += f"Samples:            {self.n_samples}\n"
        summary += f"Shape:              {self.shape}\n"
        summary += f"Sample size:        {self.sample_size}\n"
        summary += f"Key patterns:       {self.key_patterns}\n"
        if self.attrs_patterns:
            summary += f"Attr patterns:      {self.attrs_patterns}\n"
        if print_:
            print(summary)
        return summary


class ZipNumpyDataset(ZipDataset):
    """Dataset for simultaneous readouts from multiple NumPy array sources.

    `ZipNumpyDataset` allows for simultaneous readouts of multiple arrays either
    from the same `.npz` archive or from different files. It bases on `ZipDataset`
    and provides convenient factory methods for NumPy-specific use cases.
    """

    @classmethod
    def from_keys(
        cls,
        path: str | Path | Sequence[str] | Sequence[Path],
        *keys: str,
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = True,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipNumpyDataset":
        """Create a `ZipNumpyDataset` from multiple keys in the same file(s).

        Reads different arrays from the same `.npz` archive(s) in parallel,
        e.g. `'images'` and `'labels'`.

        Args:
            path: File path(s) shared by all datasets.
            *keys: Key patterns, one per dataset.
            zip_as: Return format for the combined output.
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments forwarded to `NumpyDataset`.
        """
        datasets = [
            NumpyDataset(
                path,
                key,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                dtype=dtype,
                return_as=return_as,
                **kwargs,
            )
            for key in keys
        ]
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_paths(
        cls,
        *paths: str | Path | Sequence[str] | Sequence[Path],
        keys: str | Sequence[str] = "*",
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        new_axis: bool = False,
        **kwargs,
    ) -> "ZipNumpyDataset":
        """Create a `ZipNumpyDataset` from multiple file paths.

        Use this when reading from different files in parallel.
        Each path creates a separate `NumpyDataset`.

        Args:
            *paths: File paths; each creates one dataset.
            keys: Key pattern(s) applied to all files.
            zip_as: Return format for the combined output.
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            new_axis: If `True`, each file is one sample; see `FileDataset`
                for full documentation.
            **kwargs: Additional arguments forwarded to `NumpyDataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided.")
        datasets = [
            NumpyDataset(
                path,
                keys,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                dtype=dtype,
                return_as=return_as,
                new_axis=new_axis,
                **kwargs,
            )
            for path in paths
        ]
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_named_keys(
        cls,
        path: str | Path | Sequence[str] | Sequence[Path],
        keys: dict[str, str],
        strict: bool = True,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipNumpyDataset":
        """Create a `ZipNumpyDataset` with named keys returning a dict.

        Automatically builds a dict return format from the keys of `keys`.

        Args:
            path: File path(s) shared by all datasets.
            keys: Dict mapping output names to key patterns.
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments forwarded to `NumpyDataset`.
        """
        if not keys:
            raise ValueError("At least one key must be provided.")
        datasets = [
            NumpyDataset(
                path,
                pattern,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                dtype=dtype,
                return_as=return_as,
                **kwargs,
            )
            for pattern in keys.values()
        ]
        zip_as = {name: idx for idx, name in enumerate(keys)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_named_paths(
        cls,
        paths: dict[str, str | Path | Sequence[str] | Sequence[Path]],
        keys: str | Sequence[str] = "*",
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        new_axis: bool = False,
        **kwargs,
    ) -> "ZipNumpyDataset":
        """Create a `ZipNumpyDataset` with named paths returning a dict.

        Args:
            paths: Dict mapping output names to file paths.
            keys: Key pattern(s) applied to all files.
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            new_axis: If `True`, each file is one sample; see `FileDataset`
                for full documentation.
            **kwargs: Additional arguments forwarded to `NumpyDataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided.")
        datasets = [
            NumpyDataset(
                path,
                keys,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                dtype=dtype,
                return_as=return_as,
                new_axis=new_axis,
                **kwargs,
            )
            for path in paths.values()
        ]
        zip_as = {name: idx for idx, name in enumerate(paths)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

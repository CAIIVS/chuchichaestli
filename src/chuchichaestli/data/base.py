# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base classes for datasets for single-sample returns."""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from chuchichaestli.utils import prod
from chuchichaestli.data.cache import (
    nbytes,
    serial_byte_size,
    SharedArray,
    SharedDictList,
)
import warnings
from typing import Any, Literal, NamedTuple
from collections.abc import Sequence
from types import TracebackType


__all__ = ["FileDataset", "CachingDataset", "IndexedSample", "WithIndices", "with_indices"]


DataReturnTypes = Literal["tuple", "dict"] | dict


class FileDataset(Dataset, ABC):
    """Abstract base class for file-based datasets of various types.

    Provides common file handling utilities with wildcards finders.
    Files should be memory-mapped into a `_mmap` which can represent a single
    contiguous dataset or multiple subsets that will be sequentially indexed.
    Optionally, a `_mmap_attrs` memory map (congruent to `mmap`) can be loaded
    that contains metadata.

    Required definitions:
        - `load`: load the memory map(s).
        - `close`: close the memory map(s).
    """

    FILE_EXTENSIONS: list[str] = []  # override in subclasses.

    # Attributes dropped on pickling (unpickleable handles / mmap views / caches)
    # and rebuilt in the worker by `_restore_transient`.
    _TRANSIENT_ATTRS: tuple[str, ...] = ("_mmap", "_mmap_attrs")

    def __init__(
        self,
        path: str | Path | Sequence[str] | Sequence[Path] | None = None,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        has_attrs: bool = False,
        copy_on_write: bool = False,
        sample_axis: int | None = 0,
        **kwargs,
    ):
        """Constructor.

        Args:
            path: Path to file(s) or data directory. May contain wildcards.
            dtype: Data tensor type; default: torch.float32.
            return_as: Return type; one of ['tuple', 'dict', dict, None].
                If `'dict'` or `dict`, the output will always be of type `dict`,
                if `'tuple'` and the samples have no corresponding metadata,
                the sample tensor will be returned directly.
            has_attrs: Whether to fetch attribute/metadata.
            copy_on_write: Whether to use Copy-on-Write behaviour, i.e.
                whether to copy data from memory maps/files.
            sample_axis: Which axis of a file enumerates its samples, or
                `None` if the whole file is one sample.
                - `0` (default): files of shape (N, H, W) contribute N samples
                  of shape (H, W); `ds.shape == (sum(N_i), H, W)`.
                - `None`: each file contributes 1 sample of the full file
                  shape; `ds.shape == (n_files, N, H, W)`. Use for image files.
                - `k`: samples run along axis `k`; files of shape (H, N, W)
                  with `sample_axis=1` give N samples of shape (H, W).
                  Negative values index from the end of each file's shape.
            kwargs: Additional keyword arguments for subclasses.
        """
        self.dtype = dtype
        self.return_as = return_as
        self.has_attrs = has_attrs
        if isinstance(sample_axis, bool):
            raise TypeError(
                "sample_axis must be an int or None, not a bool; pass `None` "
                "for one sample per file, or an axis index such as `0`"
            )
        self.sample_axis = sample_axis
        self.files = self.glob_path(path, self.FILE_EXTENSIONS)
        self._file_offsets: list[int] = []
        self._mmap: list[Sequence] = []
        self._mmap_attrs: list[Sequence] = []
        self.cow = copy_on_write
        if self.has_files:
            self.load(**kwargs)

    @staticmethod
    def _split_glob(
        path: str | list[str], relative: bool = False
    ) -> tuple[list[str], list[str | None]]:
        """Split path containing wildcard into root and wildcard expression.

        Args:
            path: File path or list of paths, may contain wildcards.
            relative: If True, strip leading '/' from paths.

        Returns:
            Tuple of (root_paths, wildcard_patterns).
        """
        roots: list[str] = []
        patterns: list[str | None] = []

        if isinstance(path, str) and "*" in path:
            if relative:
                path = path[1:] if path.startswith("/") else path
            components = Path(path).parts
            wildcard_idx = [i for i, c in enumerate(components) if "*" in c][0]
            roots = [str(Path().joinpath(*components[:wildcard_idx]))]
            patterns = [str(Path().joinpath(*components[wildcard_idx:]))]
            return roots, patterns
        elif isinstance(path, list | tuple):
            for p in path:
                r, pat = FileDataset._split_glob(p, relative)
                roots.extend(r)
                patterns.extend(pat)
            return roots, patterns
        return [str(path)], [None]

    @staticmethod
    def glob_path(
        path: str | Path | Sequence[str] | Sequence[Path] | None,
        extensions: Sequence[str] | None = None,
    ) -> list[Path]:
        """Glob path recursively for files with specified extensions.

        Args:
            path: Filename, path or list, can contain wildcards `*` or `**`.
                If `None`, returns an empty list.
            extensions: List of valid file extensions (e.g., ['.h5', '.hdf5']).
                If `None`, all files matching the pattern are returned.

        Returns:
            List of Path objects for matching files.
        """
        if path is None:
            return []

        files: list[Path] = []
        roots, patterns = FileDataset._split_glob(path)

        for root, pattern in zip(roots, patterns):
            if pattern is None:
                files.append(Path(root))
            else:
                matched = [f for f in Path(root).rglob(pattern) if f.is_file()]
                if extensions:
                    matched = [f for f in matched if f.suffix in extensions]
                files.extend(sorted(matched))

        files, missing = FileDataset.check_files_exist(files)
        if missing:
            warnings.warn(f"{len(missing)} files not found.")
        return files

    @staticmethod
    def check_files_exist(files: list[Path]) -> tuple[list[Path], list[Path]]:
        """Check which files exist and which are missing.

        Args:
            files: List of file paths to check.

        Returns:
            Tuple of (existing_files, missing_files).
        """
        existing = [f for f in files if f.exists()]
        missing = [f for f in files if not f.exists()]
        return existing, missing

    @staticmethod
    def validate_files(
        files: list[Path],
        extensions: list[str],
        check_exists: bool = True,
        raise_on_error: bool = True,
    ) -> list[Path]:
        """Validate that all files exist and have correct extensions.

        Args:
            files: List of file paths to validate.
            extensions: List of valid file extensions.
            check_exists: If `True`, verify that each file exists.
            raise_on_error: If `True`, raise `FileNotFoundError` for missing files
                or `ValueError` for invalid extensions.
                If `False`, filter out invalid/missing files with warnings.

        Returns:
            List of valid file paths.

        Raises:
            FileNotFoundError: If any file does not exist and `raise_on_missing`.
            ValueError: If any file has an invalid extension and `raise_on_invalid_ext`.
        """
        valid: list[Path] = []

        for f in files:
            # Check existence
            if check_exists and not f.exists():
                if raise_on_error:
                    raise FileNotFoundError(f"File not found: {f}")
                else:
                    warnings.warn(f"File not found, skipping: {f}")
                    continue

            # Check extensions
            if extensions is not None and f.suffix not in extensions:
                if raise_on_error:
                    raise ValueError(
                        f"Invalid file extension '{f.suffix}' for file: {f}. "
                        f"Expected one of: {extensions}"
                    )
                else:
                    warnings.warn(
                        f"Invalid extension '{f.suffix}' for file {f}, skipping. "
                        f"Expected one of: {extensions}"
                    )
                    continue
            valid.append(f)
        return valid

    @property
    def n_files(self) -> int:
        """Number of files in the dataset."""
        return len(self.files)

    @property
    def has_files(self) -> bool:
        """Whether the dataset has associated files or data buffers."""
        return bool(self.files) or bool(self._mmap)

    def _axis_of(self, mmap: Any) -> int:
        """Resolve `sample_axis` against one file's rank."""
        axis = self.sample_axis
        if axis is None:
            raise ValueError("the whole file is one sample; there is no sample axis")
        ndim = len(getattr(mmap, "shape", ()))
        if axis < 0:
            if not ndim:
                raise ValueError(
                    f"cannot resolve negative sample_axis={axis} without a shape"
                )
            axis += ndim
        if ndim and not (0 <= axis < ndim):
            raise ValueError(
                f"sample_axis={self.sample_axis} is out of range for a file of "
                f"shape {tuple(mmap.shape)}"
            )
        return axis

    def _axis_len(self, mmap: Any) -> int:
        """Number of samples one file contributes."""
        axis = self._axis_of(mmap)
        shape = getattr(mmap, "shape", None)
        return len(mmap) if shape is None else shape[axis]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the contiguous dataset."""
        if not self._mmap:
            return ()
        if self.sample_axis is None:
            return (self.n_files, *self._mmap[0].shape)
        total_samples = sum(self._axis_len(m) for m in self._mmap)
        first = self._mmap[0]
        axis = self._axis_of(first)
        file_shape = tuple(getattr(first, "shape", ()))
        sample_shape = file_shape[:axis] + file_shape[axis + 1 :]
        return (total_samples, *sample_shape)

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset (across all files)."""
        return self.shape[0] if self.shape else 0

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (excluding batch dimension)."""
        return self.shape[1:] if len(self.shape) >= 1 else ()

    def __len__(self) -> int:
        """Dataset length."""
        return self.n_samples

    def __str__(self) -> str:
        """Instance string."""
        name = self.__class__.__name__
        return f"{name}(#f{self.n_files}#s{self.n_samples})"

    def __repr__(self) -> str:
        """Instance representation."""
        return self.__str__()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Context manager exit."""
        self.close()
        return False

    def _build_index(self):
        """Build (cumulative) index for mapping global index to (file_index, local_index)."""
        self._file_offsets = [0]
        for m in self._mmap:
            n_samples = 1 if self.sample_axis is None else self._axis_len(m)
            self._file_offsets.append(self._file_offsets[-1] + n_samples)

    def _map_index(self, index: int) -> tuple[int, int]:
        """Map global index to (file_index, local_index).

        Args:
            index: Sample index.
        """
        if not self._file_offsets or self.n_files != (len(self._file_offsets) - 1):
            self._build_index()
        if self.sample_axis is None:
            if not (0 <= index < self.n_files):
                raise IndexError(f"Index {index} out of range")
            return index, None
        for file_idx, (ini_idx, fin_idx) in enumerate(
            zip(self._file_offsets[:-1], self._file_offsets[1:])
        ):
            if index < fin_idx:
                local_idx = index - ini_idx
                return file_idx, local_idx
        raise IndexError(f"Index {index} out of range")

    def _format_output(
        self,
        item: torch.Tensor,
        attrs: dict | Any | None,
    ) -> torch.Tensor | tuple | dict:
        """Format the output based on return_as setting.

        Args:
            item: Tensor data.
            attrs: Attributes/metadata.

        Returns:
            Formatted output.
        """
        match (self.return_as, attrs):
            case ("tuple", None):
                return item
            case ("tuple", _):
                return (item, attrs)
            case ("dict", _):
                return {"data": item, "attrs": attrs}
            case (dict() as template, _):
                keys = list(template.keys())
                match keys:
                    case [data_key, attrs_key, *_]:
                        return {data_key: item, attrs_key: attrs}
                    case [data_key]:
                        return {data_key: item, "attrs": attrs}
                    case _:
                        return {"data": item, "attrs": attrs}
            case (_, None):
                return item
            case _:
                return (item, attrs)

    def _read_item(
        self, index: int, copy: bool | None = None, flush: bool = True
    ) -> torch.Tensor:
        """Fetch tensor sample from mmap (no cache lookup).

        Args:
            index: Sample index.
            copy: Whether to copy manually from mmap; defaults to `self.cow`.
            flush: Clear memory after reading from mmap.

        Note: if subclasses have non-standard logic for accessing mmaps,
            this method should be overridden.
        """
        file_idx, local_idx = self._map_index(index)
        # Fetch sample from memory map
        copy = copy if copy is not None else not self.cow
        sample = self._get_from_mmap(file_idx, local_idx, copy=copy)
        # Flush memory-map cache
        if flush and hasattr(self._mmap[file_idx], "flush"):
            try:
                self._mmap[file_idx].flush()
            except (AttributeError, OSError):
                pass
        if isinstance(sample, torch.Tensor):
            return sample.to(self.dtype)
        return torch.from_numpy(sample).type(self.dtype)

    def _get_from_mmap(
        self, file_idx: int, local_idx: int | None, copy: bool = False
    ) -> np.ndarray:
        """Hook for reading sample from memory map.

        Override if `_mmap` requires special logic.

        Args:
            file_idx: Index of the file in self._mmap.
            local_idx: Local sample index within that file, or `None` to read
                the entire file as one sample (`sample_axis=None` mode).
            copy: Whether to manually copy from mmap.
        """
        # Fetch item
        mmap = self._mmap[file_idx]
        if local_idx is None:
            sample = mmap[:]
        else:
            axis = self._axis_of(mmap)
            # Scalar indexing on axis 0; a non-zero axis needs a mmap that takes 
            # a tuple index (numpy memmap, h5py, torch tensor).
            sample = (
                mmap[local_idx]
                if axis == 0
                else mmap[(slice(None),) * axis + (local_idx,)]
            )
        if isinstance(sample, torch.Tensor):
            return sample.clone() if copy else sample
        # Copy sample from memory map
        if copy:
            sample = sample.copy()
        # Given a memmap `fp`, `isinstance(fp, np.ndarray)` returns `True`
        if not isinstance(sample, np.ndarray):
            sample = np.asarray(sample)
        return sample

    def _read_attrs(
        self, index: int, copy: bool | None = None, flush: bool = True
    ) -> dict | Any | None:
        """Fetch attributes from mmap (no cache lookup).

        Args:
            index: Attributes index.
            copy: Whether to manually copy from mmap.
            flush: Clear memory after reading from mmap.

        Note: if subclasses have non-standard logic for accessing mmaps,
            this method should be overridden.
        """
        if not self._mmap_attrs:
            return None
        file_idx, local_idx = self._map_index(index)
        copy = copy if copy is not None else not self.cow
        attr = self._get_from_mmap_attrs(file_idx, local_idx, copy=copy)
        # Flush memory-map cache
        if flush and hasattr(self._mmap_attrs[file_idx], "flush"):
            try:
                self._mmap_attrs[file_idx].flush()
            except (AttributeError, OSError):
                pass
        return attr

    def _get_from_mmap_attrs(self, file_idx: int, local_idx: int | None, copy: bool = True):
        """Hook for reading attribute/metadata from memory map.

        Override if `_mmap_attrs` requires special logic.

        Args:
            file_idx: Index of the file in self._mmap.
            local_idx: Local sample index within that file, or `None` when
                the whole file is one sample (`sample_axis=None` mode).
            copy: Whether to manually copy from mmap.
        """
        attrs_obj = self._mmap_attrs[file_idx]
        # Whole-file mode: attrs belong to the file, not a row within it.
        if local_idx is None:
            if hasattr(attrs_obj, "items"):
                return dict(attrs_obj)
            if hasattr(attrs_obj, "keys"):
                return {k: attrs_obj[k] for k in attrs_obj.keys()}
            return attrs_obj
        # Per-sample attributes
        if hasattr(attrs_obj, "__getitem__"):
            try:
                attr = attrs_obj[local_idx]
                if isinstance(attr, torch.Tensor):
                    return attr.clone() if copy else attr
                if isinstance(attr, np.ndarray) and not attr.flags["OWNDATA"]:
                    if copy:
                        attr = attr.copy()
                return attr
            except (IndexError, TypeError, KeyError):
                pass
        # Per-group dict-like attributes
        if hasattr(attrs_obj, "items"):
            return dict(attrs_obj)
        if hasattr(attrs_obj, "keys"):
            return {k: attrs_obj[k] for k in attrs_obj.keys()}
        # Fallback: return as-is
        return attrs_obj

    def __getitem__(self, index: int) -> torch.Tensor | tuple | dict:
        """Get a sample at specified index.

        Args:
            index: Sample index.
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )
        item = self._read_item(index)
        attrs = None
        if self.has_attrs and self._mmap_attrs:
            attrs = self._read_attrs(index)
        return self._format_output(item, attrs)

    @abstractmethod
    def load(self, **kwargs):
        """Load file contents into a memory map(s) and set up resources."""
        ...

    @abstractmethod
    def close(self, **kwargs):
        """Close any open file handles and clean up resources."""
        ...

    def __getstate__(self) -> dict:
        """Drop transient/unpickleable state so the dataset survives pickling.

        Enables `DataLoader(num_workers>0)` under the `spawn`/`forkserver` start
        methods; the dropped state is rebuilt by `__setstate__` in the worker.
        """
        drop = set().union(
            *(getattr(c, "_TRANSIENT_ATTRS", ()) for c in type(self).__mro__)
        )
        return {k: v for k, v in self.__dict__.items() if k not in drop}

    def __setstate__(self, state: dict):
        """Restore config and rebuild transient state in the (worker) process."""
        self.__dict__.update(state)
        self._restore_transient()

    def _restore_transient(self):
        """Reopen handles and rebuild the memory maps (mirrors `__init__`)."""
        self._mmap = []
        self._mmap_attrs = []
        self._file_offsets = []
        if self.has_files:
            self.load()


class CachingDataset(FileDataset, ABC):
    """Base class for file-based datasets with shared memory caching support.

    Extends FileDataset with stochastic caching using SharedArray.
    All files are treated as a single contiguous dataset for caching purposes.
    The cache stores samples in shared memory, allowing efficient access across
    multiple DataLoader workers. When cache size is smaller than the dataset,
    only a subset of samples are cached (stochastic caching).

    Required definitions:
        - `load`: load the memory map(s).
        - `close`: close the memory map(s).

    The shared-memory caches (`cache`, `attrs_cache`) are picklable and re-attach
    by name in worker processes, so they are shared across `spawn`/`forkserver`
    workers (and inherited under `fork`) rather than dropped.
    """

    def __init__(
        self,
        path: str | Path | list[str] | list[Path] | None = None,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        cache: int | float | str | bool | nbytes | None = "4G",
        attrs_cache: int | float | str | bool | nbytes | None = None,
        preload: bool = False,
        sample_axis: int | None = 0,
        **kwargs,
    ):
        """Constructor.

        Args:
            path: Path to file(s) or data directory. May contain wildcards.
                If None, the dataset is assumed to be created programmatically.
            dtype: Data tensor type; default: `torch.float32`.
            return_as: Return type; one of `['tuple', 'dict', None]` or as dictionary template.
            cache: Cache size for data items (e.g., "4G", 4.0, or bytes).
            attrs_cache: Cache size for attributes/metadata.
            preload: Preload and cache the dataset.
            sample_axis: Which axis enumerates samples, or `None` for one
                sample per file; see `FileDataset` for full documentation.
            kwargs: Additional keyword arguments for subclasses.
        """
        super().__init__(
            path=path,
            dtype=dtype,
            return_as=return_as,
            sample_axis=sample_axis,
            **kwargs,
        )

        self._req_cache_size = nbytes(cache)
        self._req_attrs_cache_size = nbytes(attrs_cache)
        self.cache = SharedArray((), size=0)
        self.attrs_cache = SharedDictList(0, size=0)
        self.init_cache(self._req_cache_size, self._req_attrs_cache_size)

        if preload and self.cache:
            self.preload_cache()

    def __str__(self) -> str:
        """Instance string."""
        name = self.__class__.__name__
        cached_str = self.cache.cached_bytes.as_str() if self.cache else None
        size_str = self.cache.cache_size.as_str() if self.cache else None
        return f"{name}[{cached_str}/{size_str}](#f{self.n_files}#s{self.n_samples})"

    def __del__(self):
        """Cleanup caches when object is destroyed."""
        try:
            self.purge_cache(reset=False)
        except Exception:
            pass  # Ignore errors during cleanup

    @property
    def n_cached(self) -> int:
        """Number of cached data samples."""
        return self.cache.cached_states if self.cache else 0

    @property
    def n_cacheable(self) -> int:
        """Maximum number of samples that can fit in cache."""
        return len(self.cache) if self.cache else 0

    @property
    def cached_bytes(self) -> nbytes:
        """Byte size of currently used cache (excluding metadata cache)."""
        return self.cache.cached_bytes if self.cache else nbytes(0)

    @property
    def cached_bytes_total(self) -> nbytes:
        """Total byte size of currently used cache (including metadata cache)."""
        cbytes = self.cache.cached_bytes if self.cache else nbytes(0)
        cbytes += self.attrs_cache.cached_bytes if self.attrs_cache else nbytes(0)
        return cbytes

    @property
    def cache_size(self) -> nbytes:
        """Cache size in bytes (excluding metadata cache)."""
        return self.cache.cache_size if self.cache else nbytes(0)

    @property
    def cache_size_total(self) -> nbytes:
        """Total cache size in bytes (including metadata cache)."""
        size = self.cache.cache_size if self.cache else nbytes(0)
        size += self.attrs_cache.cache_size if self.attrs_cache else nbytes(0)
        return size

    @property
    def sample_size(self) -> nbytes:
        """Size of a single sample in bytes."""
        if not self.sample_shape:
            return nbytes(0)
        elem_size = torch.empty((), dtype=self.dtype).element_size()
        n_elements = prod(self.sample_shape)
        return nbytes(n_elements * elem_size)

    @property
    def attrs_size(self) -> nbytes:
        """Size of a single metadata element in bytes."""
        if not self._mmap_attrs:
            return nbytes(0)
        attr = self._get_from_mmap_attrs(0, 0)
        return serial_byte_size(attr)

    @property
    def serial_size(self) -> nbytes:
        """Total serialized size of the dataset."""
        return nbytes(self.n_samples * self.sample_size)

    def init_cache(
        self,
        cache: int | float | str | nbytes | None = None,
        attrs_cache: int | float | str | nbytes | None = None,
    ):
        """Initialize the shared-memory cache.

        Args:
            cache: Cache size for data items.
            attrs_cache: Cache size for attributes/metadata.
        """
        if not self.has_files or self.n_samples == 0:
            return
        self.purge_cache(reset=False)

        cache_size = nbytes(cache) if cache is not None else self._req_cache_size
        attrs_cache_size = (
            nbytes(attrs_cache)
            if attrs_cache is not None
            else self._req_attrs_cache_size
        )

        if cache_size is not None:
            self.cache = SharedArray(
                shape=self.shape,
                size=cache_size,
                dtype=self.dtype,
                allow_overwrite=True,
                verbose=False,
            )
        if attrs_cache_size is not None:
            self.attrs_cache = SharedDictList(
                n=self.n_samples,
                size=attrs_cache_size,
                slot_size=self.attrs_size,
                allow_overwrite=True,
                verbose=False,
            )

    def preload_cache(self):
        """Pre-load data and cache it."""
        if (not self.cache) and (not self.attrs_cache):
            return
        cache_slots = self.n_cacheable
        for i in range(min(len(self), cache_slots)):
            if self.cache and i not in self.cache:
                item = self._read_item(i)
                self.cache_item(i, item)

    def clear_item(self, index: int | None = None):
        """Clear single cached sample and metadata."""
        if self.cache is not None:
            self.cache.clear(index)
        if self.attrs_cache is not None:
            self.attrs_cache.clear(index)

    def purge_cache(self, reset: bool = True):
        """Purge the cache.

        Args:
            reset: If True, the cache is reinitialized after purging.
        """
        cache_size = nbytes(0)
        attrs_cache_size = nbytes(0)
        if self.cache is not None:
            cache_size += self.cache.cache_size
            self.cache.clear_allocation()
        if self.attrs_cache is not None:
            attrs_cache_size += self.attrs_cache.cache_size
            self.attrs_cache.clear_allocation()
        self.cache = None
        self.attrs_cache = None
        if reset:
            self.init_cache(cache_size, attrs_cache_size)

    def get_cached(self, index: int) -> torch.Tensor | None:
        """Get sample from cache if present.

        Args:
            index: Sample index.
        """
        if self.cache and index in self.cache:
            return self.cache[index]
        return None

    def cache_item(self, index: int, item: torch.Tensor, overwrite: bool = False):
        """Store sample tensor in cache.

        Args:
            index: Sample index.
            item: Tensor to cache.
            overwrite: If True, overwrite existing cached item.
        """
        if not self.cache:
            return
        if overwrite or index not in self.cache:
            self.cache[index] = item

    def get_cached_attrs(self, index: int) -> dict | None:
        """Get element from metadata cache if present.

        Args:
            index: Sample index.
        """
        if self.has_attrs and self.attrs_cache and index in self.attrs_cache:
            return self.attrs_cache[index]
        return None

    def cache_attrs(self, index: int, attrs: dict | Any, overwrite: bool = False):
        """Store attributes in cache.

        Args:
            index: Sample index.
            attrs: Attributes to store in the metadata cache (typically a dict).
            overwrite: If True, overwrite existing cached attribute.
        """
        if not self.attrs_cache:
            return
        if overwrite or index not in self.attrs_cache:
            self.attrs_cache[index] = attrs

    def __getitem__(self, index: int) -> torch.Tensor | tuple | dict:
        """Get a sample at specified index.

        Args:
            index: Sample index.
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )
        item = self.get_cached(index)
        if item is None:
            item = self._read_item(index)
            self.cache_item(index, item)
        attrs = self.get_cached_attrs(index)
        if self.has_attrs and attrs is None:
            attrs = self._read_attrs(index)
            self.cache_attrs(index, attrs)
        return self._format_output(item, attrs)

    def close(self):
        """Close any open file handles and purge the cache."""
        self.purge_cache(reset=False)


class IndexedSample(NamedTuple):
    """A `(index, sample)` pair produced by `WithIndices`."""

    index: int
    sample: Any


class WithIndices(Dataset):
    """Wrap a dataset so `__getitem__(i)` yields `IndexedSample(i, dataset[i])`.

    Lets a downstream `collate_fn` recover each sample's global index (e.g. to
    attach source-file provenance to a batch).
    """

    def __init__(self, dataset: Dataset):
        """Constructor.

        Args:
            dataset: Any indexable dataset.
        """
        self.dataset = dataset

    def __len__(self) -> int:
        """Length of the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> IndexedSample:
        """Return `(index, dataset[index])`."""
        return IndexedSample(index, self.dataset[index])


def with_indices(dataset: Dataset) -> WithIndices:
    """Return a `WithIndices` view of `dataset` (see `WithIndices`)."""
    return WithIndices(dataset)

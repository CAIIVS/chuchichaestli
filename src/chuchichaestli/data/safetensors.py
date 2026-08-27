# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch dataset classes for safetensors file reading."""

from pathlib import Path
import fnmatch
import torch
from safetensors import safe_open
from collections.abc import Sequence
from chuchichaestli.data.cache import nbytes
from chuchichaestli.data.base import CachingDataset, DataReturnTypes
from chuchichaestli.data.zip import ZipDataset


__all__ = ["SafetensorsDataset", "ZipSafetensorsDataset"]


class SafetensorsView:
    """Lazy, array-like view that wraps a `safe_open` handle.

    Supports integer indexing (`view[i]`) and exposes `.shape` and `.dtype`
    so that it behaves like a memory-mapped `numpy.ndarray`.

    When multiple keys are provided their sample dimensions (axis 0) are
    logically concatenated; the correct key and local offset are resolved at
    index time without materialising the whole array.
    """

    def __init__(
        self,
        handle,  # safetensors file buffer
        keys: list[str],
        key_lengths: list[int],
        sample_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self._handle = handle
        self._keys = keys
        self._key_lengths = key_lengths
        self._sample_shape = sample_shape  # shape of a single sample
        self._dtype = dtype
        self._total = sum(key_lengths)
        # cumulative offsets for key lookup
        self._offsets: list[int] = []
        acc = 0
        for length in key_lengths:
            self._offsets.append(acc)
            acc += length

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._total,) + self._sample_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return sample *idx* as a NumPy array (lazy, one slice per call)."""
        if idx < 0:
            idx += self._total
        if not (0 <= idx < self._total):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {self._total}."
            )
        # Find which key owns this index
        key_idx = 0
        for i, offset in enumerate(self._offsets):
            if i + 1 < len(self._offsets) and idx >= self._offsets[i + 1]:
                continue
            key_idx = i
            break
        local_idx = idx - self._offsets[key_idx]
        key = self._keys[key_idx]
        # safe_open lazy slice: get_slice returns a SliceView
        return self._handle.get_slice(key)[local_idx]


class SafetensorsDataset(CachingDataset):
    """Dataset for loading .safetensors files with caching features.

    Features:
    - lazy per-sample access via safetensors.safe_open
    - shared memory caching via CachingDataset
    - tensor selection with wildcard support
    - optional attribute/metadata tensor loading
    - multi-file support with automatic index fusing
    """

    FILE_EXTENSIONS = [".safetensors"]

    # safe_open handles are unpickleable
    _TRANSIENT_ATTRS = ("st_buffers",)

    def __init__(
        self,
        path: str | Path | Sequence[str] | Sequence[Path],
        keys: str | Sequence[str] = "*",
        attrs_keys: str | Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        cache: int | float | str | bool | nbytes | None = "4G",
        attrs_cache: int | float | str | bool | nbytes | None = None,
        preload: bool = False,
        **kwargs,
    ):
        """Constructor.

        Args:
            path: Path to file(s) or data directory. May contain wildcards (`*` and `**`).
            keys: Key name(s) or patterns to select data tensors.
                Use `"*"` for all keys (default).
            attrs_keys: Key name(s) or patterns for attribute/metadata tensors.
                If `None`, no attributes are loaded.
            dtype: Data tensor type; default: `torch.float32`.
            return_as: Return type; one of `['tuple', 'dict']` or a custom dict mapping.
                - `'tuple'`: Returns tuple of samples (default).
                - `'dict'`: Returns dict with key `'data'` (and
                  `'attrs'` when attributes are present).
                - `dict`: Custom mapping, e.g. `{'input': 0, 'target': 1}`.
            cache: Cache size for data items (e.g. `"4G"`, `4.0`, or bytes).
            attrs_cache: Cache size for attributes/metadata.
            preload: Preload and cache the dataset.
            kwargs: Reserved for forward-compatibility.

        Note: `sample_axis=None` (from inherited class) is not supported; each file may
            contain multiple keys with heterogeneous shapes and has no single tensor.
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
        # Open file handles and SafetensorsView objects are stored here so
        # that they can be closed explicitly.
        self.st_buffers: list = []  # list of safe_open handles

        super().__init__(
            path=path,
            dtype=dtype,
            return_as=return_as,
            cache=cache,
            attrs_cache=attrs_cache,
            preload=preload,
            copy_on_write=False,
            sample_axis=0,
            has_attrs=attrs_keys is not None,
        )

    def load(self, **kwargs):
        """Open all `.safetensors` files and build lazy views."""
        for file_path in self.files:
            handle = safe_open(str(file_path), framework="pt")
            self.st_buffers.append(handle)
            all_keys: list[str] = list(handle.keys())

            # Collect attribute keys first so they can be excluded from data keys
            attrs_matched: list[str] = []
            if self.attrs_patterns is not None:
                attrs_matched = self._match_keys(all_keys, self.attrs_patterns)
            data_keys = self._match_keys(
                [k for k in all_keys if k not in attrs_matched],
                self.key_patterns,
            )
            # Build data mmap (concatenate if multiple keys match)
            if data_keys:
                view = self._build_view(handle, data_keys, file_path)
                self._mmap.append(view)
            if self.attrs_patterns is not None:
                if attrs_matched:
                    attrs_view = self._build_view(handle, attrs_matched, file_path)
                    self._mmap_attrs.append(attrs_view)
                else:
                    self._mmap_attrs.append(None)  # keep index aligned
        self._build_index()

    @staticmethod
    def _build_view(
        handle,
        keys: list[str],
        source: Path,
    ) -> SafetensorsView:
        """Construct a `SafetensorsView` for keys in safetensors file buffer.

        Args:
            handle: An open `safetensors.safe_open` handle (`framework="pt"`).
            keys: Tensor keys to include (already matched and ordered).
            source: Source file path (used in error messages only).

        Returns:
            A :class:`_SafetensorsView` representing the concatenation of the
            selected tensors along axis 0.
        """
        # Peek shapes from the slice metadata and dtypes from a zero-length
        # slice; neither materialises any tensor data.
        slices = [handle.get_slice(k) for k in keys]
        shapes = [tuple(sl.get_shape()) for sl in slices]
        dtypes = [sl[0:0].dtype for sl in slices]
        # Validate that all selected tensors share the same sample shape and type
        ref_shape: tuple[int, ...] = shapes[0][1:]
        ref_dtype: torch.dtype = dtypes[0]
        for key, shape, dtype in zip(keys[1:], shapes[1:], dtypes[1:]):
            if shape[1:] != ref_shape:
                raise ValueError(
                    f"Tensor shapes are incompatible for concatenation in '{source}': "
                    f"key '{keys[0]}' has sample shape {ref_shape}, "
                    f"key '{key}' has sample shape {shape[1:]}"
                )
            if dtype != ref_dtype:
                raise ValueError(
                    f"Tensor dtypes are incompatible for concatenation in '{source}': "
                    f"key '{keys[0]}' has dtype {ref_dtype}, "
                    f"key '{key}' has dtype {dtype}"
                )
        key_lengths = [s[0] for s in shapes]
        return SafetensorsView(handle, keys, key_lengths, ref_shape, ref_dtype)

    @staticmethod
    def _match_keys(keys: list[str], patterns: Sequence[str]) -> list[str]:
        """Return keys matching at least one pattern (ordered, deduped).

        Args:
            keys: Candidate key names.
            patterns: matching patterns (`*` and `**` supported).
        """
        matched: list[str] = []
        for pattern in patterns:
            for key in keys:
                if fnmatch.fnmatch(key, pattern) and key not in matched:
                    matched.append(key)
        return matched

    def _restore_transient(self):
        """Reset handle list before reopening files in the worker process."""
        self.st_buffers = []
        super()._restore_transient()

    def close(self):
        """Close all open safetensors buffers and clean up resources."""
        self.st_buffers.clear()
        self._mmap.clear()
        self._mmap_attrs.clear()
        self._file_offsets.clear()
        super().close()

    @property
    def n_datasets(self) -> int:
        """Total number of tensor groups (views) across all files."""
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


class ZipSafetensorsDataset(ZipDataset):
    """Dataset for simultaneous readouts from multiple safetensors tensor sources.

    `ZipSafetensorsDataset` allows for simultaneous readouts of multiple tensors
    either from the same `.safetensors` file or from different files. It builds
    on `ZipDataset` and provides convenient factory methods for
    safetensors-specific use cases.
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
    ) -> "ZipSafetensorsDataset":
        """Create a `ZipSafetensorsDataset` from multiple sources in the same file(s).

        Reads different tensors from the same `.safetensors` file(s) in parallel,
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
            **kwargs: Additional arguments forwarded to `SafetensorsDataset`.
        """
        if not keys:
            raise ValueError("At least one dataset key must be provided.")
        datasets = [
            SafetensorsDataset(
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
        **kwargs,
    ) -> "ZipSafetensorsDataset":
        """Create a `ZipSafetensorsDataset` from multiple file paths.

        Use this when reading from different files in parallel.  Each path
        creates a separate `SafetensorsDataset`.

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
            **kwargs: Additional arguments forwarded to `SafetensorsDataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided.")
        datasets = [
            SafetensorsDataset(
                path,
                keys,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                dtype=dtype,
                return_as=return_as,
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
    ) -> "ZipSafetensorsDataset":
        """Create a `ZipSafetensorsDataset` with named keys returning a dict.

        Automatically builds a dict return format from `keys`.

        Args:
            path: File path(s) shared by all datasets.
            keys: Dict mapping output names to key patterns.
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments forwarded to `SafetensorsDataset`.
        """
        if not keys:
            raise ValueError("At least one key must be provided.")
        datasets = [
            SafetensorsDataset(
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
        **kwargs,
    ) -> "ZipSafetensorsDataset":
        """Create a `ZipSafetensorsDataset` with named paths returning a dict.

        Args:
            paths: Dict mapping output names to file paths.
            keys: Key pattern(s) applied to all files.
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments forwarded to `SafetensorsDataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided.")
        datasets = [
            SafetensorsDataset(
                path,
                keys,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                dtype=dtype,
                return_as=return_as,
                **kwargs,
            )
            for path in paths.values()
        ]
        zip_as = {name: idx for idx, name in enumerate(paths)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

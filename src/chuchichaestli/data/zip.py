# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Zip iterator for PyTorch datasets for paired readouts."""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from chuchichaestli.data.base import CachingDataset, DataReturnTypes
from chuchichaestli.data.cache import nbytes
import warnings
from typing import Any
from collections.abc import Sequence
from types import TracebackType


__all__ = ["ZipDataset"]


class ZipDataset(Dataset):
    """Dataset that zips multiple datasets together.

    Returns tuples (or dicts) of samples from multiple datasets at the same index.
    Similar to Python's built-in `zip` function, but for PyTorch datasets.
    The length is determined by the shortest dataset.
    """

    def __init__(
        self,
        *datasets: Dataset,
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = False,
    ):
        """Constructor.

        Args:
            *datasets: Various datasets to zip together.
            zip_as: Return type; one of `['tuple', 'dict', 'list']` or custom dict mapping.
                - 'tuple': Returns tuple of samples (default)
                - 'dict': Returns dict with keys '0', '1', etc.
                - dict: Custom mapping, e.g., {'input': 0, 'target': 1, 'mask': 3}
                    which maps keys to dataset indices.
            strict: If `True`, all datasets must have the same length.
                If `False`, length is determined by the shortest dataset.
        """
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided.")

        self.datasets = list(datasets)
        self.zip_as = zip_as
        self.strict = strict

        lengths = [len(d) for d in self.datasets]
        self._min_len = min(lengths)

        if strict and len(set(lengths)) > 1:
            raise ValueError(
                f"All datasets must have the same length when `strict=True`. "
                f"Got lengths: {lengths}"
            )

        if self._min_len == 0:
            warnings.warn("ZipDataset has length 0 (one or more datasets are empty)")

    @classmethod
    def from_paths(
        cls,
        *paths: Sequence[str] | Sequence[Path],
        dataset_cls: type[CachingDataset],
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipDataset":
        """Create ZipDatasets from multiple file paths with caching.

        Args:
            *paths: File paths for each dataset. Each path can be a single file,
                directory, or pattern with wildcards `*` or `**`.
            dataset_cls: CachingDataset subclass to instantiate (e.g., HDF5Dataset).
            zip_as: Return format for ZipDataset; one of
                `['tuple', 'dict', 'list']` or custom dict mapping.
            strict: If `True`, all datasets must have the same length.
                If `False`, length is determined by the shortest dataset.
            cache: Cache size for each dataset's samples.
            attrs_cache: Cache size for each dataset's attributes/metadata.
            preload: Whether to preload and cache all datasets.
            dtype: Data tensor type for all datasets.
            return_as: Return format for individual datasets.
            **kwargs: Additional keyword arguments passed to `dataset_cls`.
        """
        if not paths:
            raise ValueError("At least one path must be provided")
        datasets = []
        for path in paths:
            dataset = dataset_cls(
                path=path,
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            datasets.append(dataset)
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_named_paths(
        cls,
        paths: dict[str, str | Path | Sequence[str] | Sequence[Path]],
        dataset_cls: type[CachingDataset],
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipDataset":
        """Create ZipDataset from named paths with automatic dict return format.

        Args:
            paths: Dictionary mapping names to file paths.
                Keys will be used in the return dict.
            dataset_cls: CachingDataset subclass to instantiate (e.g., HDF5Dataset).
            strict: If `True`, all datasets must have the same length.
                If `False`, length is determined by the shortest dataset.
            cache: Cache size for each dataset's samples.
            attrs_cache: Cache size for each dataset's attributes/metadata.
            preload: Whether to preload and cache all datasets.
            dtype: Data tensor type for all datasets.
            return_as: Return format for individual datasets.
            **kwargs: Additional keyword arguments passed to `dataset_cls`.
        """
        if not paths:
            raise ValueError("At least one named path must be provided")
        # Create return_as mapping from names to indices
        names = list(paths.keys())
        zip_as = {name: idx for idx, name in enumerate(names)}
        # Create datasets in order
        datasets = []
        for name in names:
            dataset = dataset_cls(
                path=paths[name],
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            datasets.append(dataset)
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @property
    def n_datasets(self) -> int:
        """Number of datasets being zipped."""
        return len(self.datasets)

    def __len__(self) -> int:
        """Length of the zipped dataset (minimum of all dataset lengths)."""
        return self._min_len

    def __str__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        return f"{name}(n={self.n_datasets}, len={len(self)})"

    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()

    def __enter__(self):
        """Context manager entry - enters all datasets."""
        for dataset in self.datasets:
            if hasattr(dataset, "__enter__"):
                dataset.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Context manager exit - exits all datasets."""
        for dataset in self.datasets:
            if hasattr(dataset, "__exit__"):
                dataset.__exit__(exc_type, exc_val, exc_tb)
        return False

    def _format_output(self, items: list[Any]) -> tuple | dict:
        """Format the output based on return_as setting.

        Args:
            items: List of samples from each dataset.
        """
        match self.zip_as:
            case "tuple":
                return tuple(items)
            case "dict":
                return {i: item for i, item in enumerate(items)}
            case dict() as template:
                # Custom mapping from keys to dataset indices
                result = {}
                for key, idx in template.items():
                    if isinstance(idx, int):
                        if 0 <= idx < len(items):
                            result[key] = items[idx]
                        else:
                            warnings.warn(
                                f"Dataset index {idx} out of range for key '{key}'. "
                                f"Valid range: [0, {len(items)})"
                            )
                    else:
                        warnings.warn(
                            f"Invalid dataset index type for key '{key}': {type(idx)}. "
                            f"Expected int."
                        )
                return result
            case _:
                return tuple(items)

    def __getitem__(self, index: int) -> tuple | dict:
        """Get samples at specified index from all datasets.

        Args:
            index: Sample index.
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )
        items = [dataset[index] for dataset in self.datasets]
        return self._format_output(items)

    def close(self):
        """Close all datasets if they support closing."""
        for dataset in self.datasets:
            if hasattr(dataset, "close") and callable(dataset.close):
                dataset.close()

    def purge_cache(self, reset: bool = True):
        """Purge caches of all datasets if they support caching.

        Args:
            reset: If True, reinitialize caches after purging.
        """
        for dataset in self.datasets:
            if hasattr(dataset, "purge_cache") and callable(dataset.purge_cache):
                dataset.purge_cache(reset=reset)

    @property
    def n_cached(self) -> int:
        """Total number of cached samples across all datasets."""
        total = 0
        for dataset in self.datasets:
            if hasattr(dataset, "n_cached"):
                total += dataset.n_cached
        return total

    @property
    def cached_bytes_total(self) -> nbytes:
        """Total cached bytes across all datasets."""
        total = nbytes(0)
        for dataset in self.datasets:
            if hasattr(dataset, "cached_bytes_total"):
                total += dataset.cached_bytes_total
            elif hasattr(dataset, "cached_bytes"):
                total += dataset.cached_bytes
        return total

    @property
    def cache_size_total(self) -> nbytes:
        """Total cache size across all datasets."""
        total = nbytes(0)
        for dataset in self.datasets:
            if hasattr(dataset, "cache_size_total"):
                total += dataset.cache_size_total
            elif hasattr(dataset, "cache_size"):
                total += dataset.cache_size
        return total

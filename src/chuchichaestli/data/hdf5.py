# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch dataset classes for HDF5 file reading."""

from pathlib import Path
import fnmatch
import tempfile
import h5py
import torch
import numpy as np
from collections.abc import Sequence
from chuchichaestli.data.cache import nbytes
from chuchichaestli.data.base import CachingDataset, DataReturnTypes
from chuchichaestli.data.zip import ZipDataset


__all__ = ["HDF5Dataset", "ZipHDF5Dataset"]


class HDF5Dataset(CachingDataset):
    """Dataset for loading HDF5 groups with (sto)caching features.

    Features:
    - memory-mapped HDF5 file access
    - shared memory caching via CachingDataset
    - HDF5 group selection with wildcard support
    - optional attribute/metadata loading
    - multi-file support with automatic index fusing
    """

    FILE_EXTENSIONS = [".hdf", ".h5", ".hdf5", ".he5"]

    def __init__(
        self,
        path: str | Path | Sequence[str] | Sequence[Path],
        groups: str | Sequence[str] = "*",
        attrs_groups: str | Sequence[str] | None = None,
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
            groups: HDF5 group path(s) or patterns for datasets.
                Use `"*"` for all groups, `"path/*"` for subgroups, etc.
            attrs_groups: HDF5 group path(s) or patterns for attributes/metadata.
                If `None`, no attributes are loaded.
            dtype: Data tensor type; default: `torch.float32`.
            return_as: Return type; one of `['tuple', 'dict', 'list']` or custom dict mapping.
                - `'tuple'`: Returns tuple of samples (default)
                - `'dict'`: Returns dict with keys `'0'`, `'1'`, etc.
                - `dict`: Custom mapping, e.g., `{'input': 0, 'target': 1, 'mask': 3}`
                    which maps keys to dataset indices.
            cache: Cache size for data items (e.g., `"4G"`, `4.0`, or bytes).
            attrs_cache: Cache size for attributes/metadata.
            preload: Preload and cache the dataset.
            kwargs: Additional arguments for h5py.File (e.g., `libver='latest'`).
        """
        # Keyword arguments for `h5py.File` instantiation
        self.h5_kwargs = kwargs
        self.h5_kwargs.setdefault("libver", "latest")
        # Initialize HDF5 lists
        self.h5_buffers: list[h5py.File] = []
        self.h5_datasets: list[list[h5py.Dataset]] = []
        self.h5_attrs: list[list[h5py.AttributeManager | h5py.Dataset]] = []
        self._virt_files: list[Path] = []
        # HDF5 group patterns for data/metadata
        self.group_patterns = (groups,) if isinstance(groups, str) else tuple(groups)
        self.attrs_patterns = None
        if attrs_groups is not None:
            self.attrs_patterns = (
                (attrs_groups,)
                if isinstance(attrs_groups, str)
                else tuple(attrs_groups)
            )
        # Initialize CachingDataset(FileDataset)
        super().__init__(
            path=path,
            dtype=dtype,
            return_as=return_as,
            cache=cache,
            attrs_cache=attrs_cache,
            preload=preload,
            copy_on_write=True,
            has_attrs=attrs_groups is not None,
            new_axis=False,
        )

    def load(self, **kwargs):
        """Load HDF5 files and locate datasets.

        Open all HDF5 files, find datasets matching group patterns, and build
        the index for sequential access across files.
        """
        for file_path in self.files:
            h5_file = h5py.File(file_path, "r", **self.h5_kwargs)
            self.h5_buffers.append(h5_file)
            # Look up HDF5 datasets
            datasets = self._find_datasets(h5_file, self.group_patterns)
            self.h5_datasets.append(datasets)
            # Memory-map data
            if datasets:
                if len(datasets) == 1:
                    self._mmap.append(datasets[0])
                else:
                    virt_h5file, virt_path = self._create_vds(datasets, file_path)
                    self.h5_buffers.append(virt_h5file)
                    self._virt_files.append(virt_path)
                    self._mmap.append(virt_h5file["data"])
            # Load metadata
            if self.attrs_patterns:
                attrs = self._find_attrs(h5_file, self.attrs_patterns)
                self.h5_attrs.append(attrs)
                if attrs:
                    self._mmap_attrs.append(attrs[0])
        self._build_index()

    def _get_from_mmap_attrs(self, file_idx, local_idx, copy: bool = True):
        """Hook for reading attribute/metadata from memory map.

        Args:
            file_idx: Index of the file in self._mmap.
            local_idx: Local sample index within that file.
            copy: Whether to manually copy from mmap_attrs.
        """
        attr_obj = self._mmap_attrs[file_idx]
        if isinstance(attr_obj, h5py.AttributeManager):
            # Per-group attributes
            attrs_dict = {}
            for key in attr_obj.keys():
                value = attr_obj[key]
                # Convert for JSON serialization
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        attrs_dict[key] = value.item()
                    else:
                        attrs_dict[key] = value.tolist()
                else:
                    attrs_dict[key] = value
            return attrs_dict
        # for h5py.Dataset, base class behaviour is fine
        return super()._get_from_mmap_attrs(file_idx, local_idx, copy=copy)

    @staticmethod
    def _create_vds(
        datasets: list[h5py.Dataset],
        source_file: Path,
    ) -> tuple[h5py.Dataset, Path]:
        """Create a virtual dataset layout that concatenates multiple datasets.

        Args:
            datasets: List of h5py.Dataset objects to concatenate.
            source_file: Path to the source HDF5 file.

        Returns:
            Tuple of a virtually stitched `h5py.Dataset` instance and a file
            path holding the temporary virtual file.
        """
        if not datasets:
            raise ValueError("At least one dataset required.")
        # Verify shape compatibility
        ref_shape = datasets[0].shape[1:]
        ref_dtype = datasets[0].dtype
        for i, ds in enumerate(datasets[1:], start=1):
            if ds.shape[1:] != ref_shape:
                raise ValueError(
                    f"Dataset shapes are incompatible for concatenation: "
                    f"dataset 0 has shape {datasets[0].shape}, "
                    f"dataset {i} has shape {ds.shape}"
                )
            if ds.dtype != ref_dtype:
                raise ValueError(
                    f"Dataset dtypes are incompatible for concatenation: "
                    f"dataset 0 has dtype {ref_dtype}, "
                    f"dataset {i} has dtype {ds.dtype}"
                )
        # Create virtual layout
        total_length = sum(len(ds) for ds in datasets)
        layout_shape = (total_length,) + ref_shape
        layout = h5py.VirtualLayout(shape=layout_shape, dtype=ref_dtype)
        # Map each source dataset to the virtual layout
        offset = 0
        for ds in datasets:
            source = h5py.VirtualSource(source_file, ds.name, shape=ds.shape)
            layout[offset : offset + len(ds)] = source
            offset += len(ds)
        # Create temporary file for the virtual dataset
        vds_file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".h5", delete=False)
        vds_path = Path(vds_file.name)
        vds_file.close()
        # Create virtual dataset
        with h5py.File(vds_path, "w", libver="latest") as f:
            f.create_virtual_dataset("data", layout, fillvalue=0)
        vds_h5 = h5py.File(vds_path, "r", libver="latest")
        return vds_h5, vds_path

    def _find_datasets(
        self, h5_file: h5py.File, patterns: Sequence[str]
    ) -> list[h5py.Dataset]:
        """Find datasets in HDF5 file matching patterns.

        Args:
            h5_file: Opened HDF5 file buffer.
            patterns: Group path patterns (supports * and **).
        """
        datasets: list[h5py.Dataset] = []
        # Construct group path tree of datasets
        paths: list[str] = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append(name)

        h5_file.visititems(visitor)
        # Match patterns
        for pattern in patterns:
            for path in paths:
                if fnmatch.fnmatch(path, pattern):
                    obj = h5_file[path]
                    if isinstance(obj, h5py.Dataset) and obj not in datasets:
                        datasets.append(obj)
        return datasets

    def _find_attrs(
        self, h5_file: h5py.File, patterns: Sequence[str]
    ) -> list[h5py.AttributeManager | h5py.Dataset]:
        """Find attributes or metadata datasets matching patterns.

        Args:
            h5_file: Open HDF5 file handle.
            patterns: Group path patterns for attributes.

        Returns:
            List of attribute managers or metadata datasets.
        """
        attrs: list[h5py.AttributeManager | h5py.Dataset] = []
        # Construct group path tree
        paths: list[str] = []

        def visitor(name, obj):
            paths.append(name)

        h5_file.visititems(visitor)

        # Match patterns and get metadata
        for pattern in patterns:
            for path in paths:
                if fnmatch.fnmatch(path, pattern):
                    if path in h5_file:
                        obj = h5_file[path]
                        # Check if group has attributes
                        if isinstance(obj, h5py.Group) and len(obj.attrs) > 0:
                            if obj.attrs not in attrs:
                                attrs.append(obj.attrs)
                        # Or a dataset that serves as metadata
                        elif isinstance(obj, h5py.Dataset):
                            is_data_ds = any(
                                obj == ds
                                for ds in self.h5_datasets[-1]
                                if self.h5_datasets
                            )
                            if not is_data_ds and obj not in attrs:
                                attrs.append(obj)
        return attrs

    def close(self):
        """Close HDF5 files and clean up resources."""
        # Close all HDF5 files
        for h5_buffer in self.h5_buffers:
            try:
                h5_buffer.close()
            except Exception:
                pass

        # Remove temporary virtual-dataset files (created with delete=False)
        for virt_path in self._virt_files:
            try:
                virt_path.unlink(missing_ok=True)
            except OSError:
                pass

        # Clear references
        self.h5_buffers.clear()
        self.h5_datasets.clear()
        self.h5_attrs.clear()
        self._mmap.clear()
        self._mmap_attrs.clear()
        self._virt_files.clear()
        self._file_offsets.clear()

        # Purge caches
        super().close()

    @property
    def n_datasets(self) -> int:
        """Total number of datasets across all files."""
        return sum(len(ds_list) for ds_list in self.h5_datasets)

    @property
    def dataset_groups(self) -> list[list[str]]:
        """Group paths of selected datasets for each file."""
        return [[ds.name for ds in ds_list] for ds_list in self.h5_datasets]

    @property
    def attr_groups(self) -> list[list[str]]:
        """Group paths of selected attributes for each file."""
        result = []
        for attrs_list in self.h5_attrs:
            paths = []
            for attr_obj in attrs_list:
                if isinstance(attr_obj, h5py.AttributeManager):
                    paths.append(h5py.h5i.get_name(attr_obj._id).decode())
                elif isinstance(attr_obj, h5py.Dataset):
                    paths.append(attr_obj.name)
            result.append(paths)
        return result

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
        summary += f"Group patterns:     {self.group_patterns}\n"
        if self.h5_datasets:
            summary += f"Dataset groups:     {self.dataset_groups}\n"
        if self.attrs_patterns:
            summary += f"Attr patterns:      {self.attrs_patterns}\n"
            if self.h5_attrs:
                summary += f"Attr groups:        {self.attr_groups}\n"
        if print_:
            print(summary)
        return summary


class ZipHDF5Dataset(ZipDataset):
    """Dataset for simultaneous readouts from multiple HDF5 groups.

    `ZipHDF5Dataset` allows for simultaneous readouts of multiple HDF5 groups
    either from the same file or different files. It bases on `ZipDataset` and
    provides convenient factory methods for HDF5-specific use cases.
    """

    @classmethod
    def from_groups(
        cls,
        path: str | Path | Sequence[str] | Sequence[Path],
        *groups: str,
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = True,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipHDF5Dataset":
        """Create ZipHDF5Dataset from multiple groups in the same file(s).

        This is the most common use case: reading different groups from the
        same HDF5 file(s) in parallel (e.g., data and labels/masks).

        Args:
            path: HDF5 file path(s). All datasets will use the same file(s).
            groups: List of group patterns, one for each dataset.
                Each group pattern creates a separate HDF5Dataset.
            zip_as: Return format for combined output:
                - 'tuple': (dataset0, dataset1, ...)
                - 'dict': {0: dataset0, 1: dataset1, ...}
                - dict: Custom mapping like {'data': 0, 'labels': 1}
            strict: If True, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
                If None, returns raw tensors (no tuple wrapping).
            **kwargs: Additional arguments for `HDF5Dataset`.
        """
        datasets = []
        for group in groups:
            dataset = HDF5Dataset(
                path=path,
                groups=group,
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
    def from_paths(
        cls,
        *paths: str | Path | Sequence[str] | Sequence[Path],
        groups: str | Sequence[str] = "*",
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipHDF5Dataset":
        """Create ZipHDF5Dataset from multiple file paths.

        Use this when you want to read from different files in parallel.
        Each path creates a separate HDF5Dataset.

        Args:
            *paths: HDF5 file paths. Each path creates one dataset.
            groups: Group pattern(s) applied to all files.
                Can be a single pattern or sequence.
            zip_as: Return format for combined output:
                - 'tuple': (dataset0, dataset1, ...)
                - 'dict': {0: dataset0, 1: dataset1, ...}
                - dict: Custom mapping like {'data': 0, 'labels': 1}
            strict: If True, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments for `HDF5Dataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided")
        datasets = []
        for path in paths:
            dataset = HDF5Dataset(
                path=path,
                groups=groups,
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
    def from_named_groups(
        cls,
        path: str | Path | Sequence[str] | Sequence[Path],
        groups: dict[str, str],
        strict: bool = True,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipHDF5Dataset":
        """Create ZipHDF5Dataset with named groups returning a dict.

        This is a convenience method that automatically creates a dict
        return format based on the keys in the groups dictionary.
        Applies to the same use case as `from_groups`.

        Args:
            path: HDF5 file path(s).
            groups: Dict mapping output names to group patterns.
                Keys will be used in the return dict.
            strict: If True, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments for `HDF5Dataset`.
        """
        if not groups:
            raise ValueError("At least one group must be provided")
        # Create datasets for each named group
        datasets = []
        group_names = list(groups.keys())
        group_patterns = list(groups.values())
        for pattern in group_patterns:
            dataset = HDF5Dataset(
                path=path,
                groups=pattern,
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            datasets.append(dataset)
        # Create return mapping: name -> dataset index
        zip_as = {name: idx for idx, name in enumerate(group_names)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_named_paths(
        cls,
        paths: dict[str, str | Path | Sequence[str] | Sequence[Path]],
        groups: str | Sequence[str] = "*",
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipHDF5Dataset":
        """Create ZipHDF5Dataset with named paths returning a dict.

        Applies to the same use case as `from_paths`.

        Args:
            paths: Dict mapping output names to file paths.
            groups: Group pattern(s) applied to all files.
            strict: If True, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets.
            dtype: PyTorch data type.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments for `HDF5Dataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided")

        # Create datasets for each named path
        datasets = []
        path_names = list(paths.keys())
        path_values = list(paths.values())

        for path in path_values:
            dataset = HDF5Dataset(
                path=path,
                groups=groups,
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            datasets.append(dataset)

        # Create return mapping: name -> dataset index
        zip_as = {name: idx for idx, name in enumerate(path_names)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_tuples(
        cls,
        *pairs: tuple[
            str | Path | Sequence[str] | Sequence[Path],
            str | Sequence[str],
        ],
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = True,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipHDF5Dataset":
        """Create ZipHDF5Dataset from (path, group) pairs.

        This method applies to the most general usecase: reading separate
        files and groups in parallel, for paired/tupled datasets.

        Args:
            *pairs: `(path, group)` tuples. Each tuple independently
                specifies the HDF5 file(s) and the group pattern for one
                dataset slot.
            zip_as: Return format for combined output:
                - 'tuple': (dataset0, dataset1, ...)
                - 'dict': {0: dataset0, 1: dataset1, ...}
                - dict: Custom mapping like {'data': 0, 'labels': 1}
            strict: If `True`, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets into memory.
            dtype: PyTorch data type for all datasets.
            return_as: Return format for individual datasets.
            **kwargs: Additional arguments for `HDF5Dataset`.
        """
        if not pairs:
            raise ValueError("At least one (path, group) pair must be provided")
        datasets = []
        for path, group in pairs:
            dataset = HDF5Dataset(
                path=path,
                groups=group,
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
    def from_named_tuples(
        cls,
        pairs: dict[
            str,
            tuple[
                str | Path | Sequence[str] | Sequence[Path],
                str | Sequence[str],
            ],
        ],
        strict: bool = True,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipHDF5Dataset":
        """Create ZipHDF5Dataset from named (path, group) pairs.

        Named variant of :meth:`from_tuples`. Each key in ``pairs`` becomes
        the corresponding key in the returned sample dict, so no explicit
        ``zip_as`` mapping is required.

        Args:
            pairs: Dict mapping output names to ``(path, group)`` tuples.
                Keys become the keys of the returned sample dict; values are
                ``(path, group)`` tuples as accepted by :meth:`from_tuples`.
            strict: If ``True``, all datasets must have the same length.
            cache: Cache size for each dataset.
            attrs_cache: Attribute cache size for each dataset.
            preload: Whether to preload all datasets into memory.
            dtype: PyTorch data type for all datasets.
            return_as: Return format for individual datasets.
            **kwargs: Additional keyword arguments forwarded to
                ``HDF5Dataset``.

        Raises:
            ValueError: If ``pairs`` is empty.

        Example::

            ds = ZipHDF5Dataset.from_named_tuples(
                {
                    "image":  ("train_images.h5",  "data/images"),
                    "mask":   ("train_labels.h5",  "annotations/masks"),
                    "meta":   ("train_meta.h5",    "metadata/info"),
                },
                strict=True,
            )
            sample = ds[0]          # {"image": ..., "mask": ..., "meta": ...}
        """
        if not pairs:
            raise ValueError("At least one (path, group) pair must be provided")
        names = list(pairs.keys())
        datasets = []
        for name in names:
            path, group = pairs[name]
            dataset = HDF5Dataset(
                path=path,
                groups=group,
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            datasets.append(dataset)
        zip_as = {name: idx for idx, name in enumerate(names)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

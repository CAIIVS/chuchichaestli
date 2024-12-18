"""PyTorch dataset classes.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

from pathlib import Path
import fnmatch
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from chuchichaestli.data.cache import (
    estimate_byte_size,
    SharedArray,
    SharedDictList,
)
from typing import Any
from collections.abc import Callable


class HDF5Dataset(Dataset):
    """Dataset for loading HDF5 frames (with optional shared memory caching)."""

    def __init__(
        self,
        path: str | Path | list[str | Path],
        file_key: str = "**/*.hdf5",
        groups: str | tuple[str, ...] | None = None,
        sort_key: Callable | None = None,
        meta_groups: str | tuple[str, ...] | None = None,
        scheme: str | None = None,
        dim: int = 0,
        n_channels: int | None = None,
        dtype: torch.dtype = torch.float32,
        collate: bool = True,
        preload: bool = False,
        cache: int | float | str | bool | None = "4G",
        meta_cache: int | float | str | bool | None = "64M",
        **kwargs,
    ):
        """Constructor.

        Args:
          path: Path to a file, files, or data directory which contains the HDF5 files (recursively)
          file_key: Key to filter particular files (if path is a directory)
          groups: Filter pattern for HDF5 groups containing datasets
          sort_key: Sorting key function for the list of HDF5 groups
          meta_groups: Filter pattern for HDF5 attributes containing metadata
          scheme: HDF5 metadata loading scheme how HDF5 groups for attributes (metadata) map
            onto HDF5 groups for datasets; can be
            - 'analog': HDF5 datasets and attributes are from the same HDF5 groups
            - 'bijective': HDF5 datasets and attributes are from parallel HDF5 groups
            - 'surjective': HDF5 datasets have per-sample attributes at multiple HDF5 groups
            - 'collective': HDF5 datasets have all a single HDF5 group of attributes
          dim: Dimension along which the HDF5 dataset indexing is performed.
          n_channels: Number of channels of the data (if None, a sample is automatically ).
          dtype: Data format; default: torch.float32
          collate: Collate multiple files into a single continuous dataset,
            otherwise __getitem__ pulls samples from each file.
          preload: Preload and cache the dataset.
          cache: If not None or False, memory is allocated for stochastic caching of the tensors.
          meta_cache: If not None or False, memory is allocated for stochastic caching of the metadata.
          kwargs: Keyword arguments for h5py.File.
            See https://docs.h5py.org/en/stable/high/file.html#h5py.File
        """
        if isinstance(path, str | Path) and Path(path).is_file() and Path(path).suffix == ".hdf5":
            self.files = [Path(path)]
        elif isinstance(path, list | tuple):
            self.files = [Path(p) for p in path]
        else:
            self.files = sorted([f for f in Path(path).rglob(file_key) if f.is_file()])
        self.frame_args = kwargs

        self.groups: tuple[str, ...]
        self.meta_groups: tuple[str, ...] | None
        if groups is None:
            self.groups = ("**/images*",)
        else:
            self.groups = (groups,) if isinstance(groups, str) else groups
        if meta_groups is None:
            self.meta_groups = None
            self.scheme = "analog" if scheme is None else scheme
        else:
            self.meta_groups = (
                (meta_groups,) if isinstance(meta_groups, str) else meta_groups
            )
            self.scheme = "surjective" if scheme is None else scheme
        if sort_key is not None:
            self.sort_key = sort_key
        else:
            self.sort_key = self.default_sort_key
        self.dim = dim
        self.n_channels = n_channels
        self.dtype = dtype
        self.collate = collate
        self.preload = preload

        # Temporary error warning...
        if self.scheme in ["bijective", "collective"]:
            raise NotImplementedError(f"{self.scheme} is not implemented yet.")

        self.load_frame(**self.frame_args)
        self.make_index(dim=self.dim, scheme=self.scheme, collate=self.collate)
        self.cache: tuple[SharedArray | None, SharedDictList | None] = (None, None)
        if cache and cache is not None:
            if isinstance(cache, bool):
                cache = "4G"
            if isinstance(cache, bool):
                meta_cache = "64M"
            self.cache = (
                SharedArray(shape=self.shape, size=cache, dtype=self.dtype),
                SharedDictList(
                    n=self.shape[0], slot_size=self.byte_size[1], size=meta_cache
                ),
            )
        if self.preload and self.cache[0] is not None:
            for i in range(len(self)):
                self[i]

    @staticmethod
    def default_sort_key(x: Any) -> int:
        """Default sort key for HDF5 groups."""
        return int(x.split("/")[-1]) if x.split("/")[-1].isdigit() else 0

    def load_frame(self, **kwargs) -> list[h5py.File] | None:
        """Load HDF5 file instances."""
        if self.files:
            self._frame: list[h5py.File] = [
                h5py.File(f, "r", **kwargs) for f in self.files
            ]
            return self._frame
        return None

    @property
    def frame(self) -> list[h5py.File]:
        """Lazy-loading list of HDF5 file instances."""
        if not hasattr(self, "_frame"):
            self.load_frame(**self.frame_args)
        return self._frame

    @property
    def frame_size(self) -> list[int]:
        """Dataset sizes per frame and group"""
        if not hasattr(self, "_frame_size"):
            self.make_index(dim=self.dim, scheme=self.scheme, collate=self.collate)
        return self._frame_size

    @property
    def frame_structure(self) -> dict[int, list[str]]:
        """Fetch unfiltered frame structure (HDF5 Groups) of each frame (HDF5 File instance)."""
        sort_key = self.sort_key
        if not hasattr(self, "_frame_structure"):
            tree: dict[int, list[str]] = {}
            for i, f in enumerate(self.frame):
                tree[i] = []
                f.visit(tree[i].append)
            self._frame_structure = {
                i: sorted(v, key=sort_key) for i, v in tree.items()
            }
        return self._frame_structure

    def _map_analog(
        self, frame_index: int, keys: list[str], dim: int = 0
    ) -> dict[int, tuple[str, str]]:
        """Index datasets using the 'analog' mapping scheme.

        Returns:
          mapping: A dictionary that maps continuous indices to HDF5 groups for
            HDF5 dataset samples and corresponding HDF5 attributes, e.g.
            mapping[42] -> (path/to/images, path/to/metadata_42)
            for which the usage is:

        Example:
          >>> ds = HDF5Dataset("./data", meta_groups="**/metadata/*", scheme="surjective")
          >>> frame = ds.frame[0]
          >>> keys = ["/path/to/dataset"]
          >>> attr_keys = ["/path/to/metadata"]
          >>> mapping = ds._map_surjective(0, keys, attr_keys)[0]
          >>> sample_42 = ds.frame[mapping[42][0]][42]
          >>> metadata_42 = ds.frame[mapping[42][1]].attrs
        """
        mapping: dict[int, tuple[str, str]] = {}
        total_samples = 0
        for key in keys:
            group = self.frame[frame_index][key]
            if isinstance(group, h5py._hl.dataset.Dataset):
                # assume single images if shpae ~ (N, N)
                if len(group.shape) == 2 and group.shape[0] == group.shape[1]:
                    iN = 1
                else:
                    iN = group.shape[dim]
                self._frame_size[frame_index] += iN
                total_samples += iN
                for i in range(total_samples - iN, total_samples):
                    mapping[i] = (key, key)
        return mapping

    def _map_bijective(
        self, frame_index: int, ds_keys: list[str], attr_keys: list[str], dim: int = 0
    ):
        """Index datasets using the 'bijective' mapping scheme."""
        NotImplemented

    def _map_surjective(
        self, frame_index: int, ds_keys: list[str], attr_keys: list[str], dim: int = 0
    ) -> dict[int, tuple[str, str]]:
        """Index datasets using the 'surjective' mapping scheme.

        Returns:
          mapping: A dictionary that maps continuous indices to HDF5 groups for
            HDF5 dataset samples and corresponding HDF5 attributes, e.g.
            mapping[42] -> (path/to/images_with_42, path/to/metadata_42).

        Example:
          >>> ds = HDF5Dataset("./data", meta_groups="**/metadata/*", scheme="surjective")
          >>> frame = ds.frame[0]
          >>> keys = ["/path/to/dataset"]
          >>> attr_keys = ["/path/to/metadata"]
          >>> mapping = ds._map_surjective(0, keys, attr_keys)[0]
          >>> sample_42 = ds.frame[mapping[42][0]][42]
          >>> metadata_42 = ds.frame[mapping[42][1]].attrs
        """
        mapping: dict[int, tuple[str, str]] = {}
        total_samples = 0
        for key in ds_keys:
            group = self.frame[frame_index][key]
            if isinstance(group, h5py._hl.dataset.Dataset):
                if len(group.shape) == 2 and group.shape[0] == group.shape[1]:
                    iN = 1
                else:
                    iN = group.shape[dim]
                self._frame_size[frame_index] += iN
                total_samples += iN
                if len(attr_keys) != iN:
                    raise ValueError(
                        f"Mismatch in dataset elements ({iN} at {dim=}) and "
                        f"attribute groups ({len(attr_keys)}) for surjective index mapping scheme."
                    )
                for i in range(total_samples - iN, total_samples):
                    mapping[i] = (key, attr_keys[i])
        return mapping

    def _map_collective(
        self, frame_index: int, ds_keys: list[str], attr_keys: list[str], dim: int = 0
    ):
        """Index datasets using the 'collective' mapping scheme."""
        NotImplemented

    def make_index(
        self, dim: int = 0, scheme: str | None = None, collate: bool | None = None
    ):
        """Index the frames (HDF5 File objects) for individual samples.

        The index map keys (indices) point to a list of tuples as follows:
          index -> [(frame index, group index, attribute key), ...]

        Args:
          dim: Dimension across which the datasets are stacked
          scheme: HDF5 metadata loading scheme how HDF5 groups for attributes (metadata)
            map onto HDF5 groups for datasets; can be
            - 'analog': HDF5 datasets and attributes are at the same HDF5 groups
            - 'bijective': HDF5 datasets and attributes are at parallel HDF5 groups
            - 'surjective': HDF5 datasets have per-sample attributes at multiple HDF5 groupse
            - 'collective': HDF5 datasets have all a single HDF5 group of attributes
          collate: Collate multiple files into a single continuous dataset,
            otherwise __getitem__ pulls samples from each file.
        """
        if scheme is not None:
            self.scheme = scheme
        if collate is not None:
            self.collate = collate
        # filter the frame structures for group keys pointing to datasets and metadata
        groups = {}
        meta_groups = {}
        for i_frame, fnlist in self.frame_structure.items():
            data_keys = [f for g in self.groups for f in fnmatch.filter(fnlist, g)]
            groups[i_frame] = data_keys
            if self.meta_groups:
                meta_keys = [
                    f for g in self.meta_groups for f in fnmatch.filter(fnlist, g)
                ]
                meta_groups[i_frame] = meta_keys
        # map out dataset(s) and metadata
        loc: dict[int, dict] = {}
        self._frame_size: list[int] = [0 for i_frame in groups.keys()]
        for i_frame, data_keys in groups.items():
            if i_frame in meta_groups:
                meta_keys = meta_groups[i_frame]
                match self.scheme:
                    case "analog":
                        mapping = self._map_analog(i_frame, data_keys, dim=dim)
                    case "bijective":
                        mapping = self._map_bijective(
                            i_frame, data_keys, meta_keys, dim=dim
                        )
                    case "surjective":
                        mapping = self._map_surjective(
                            i_frame, data_keys, meta_keys, dim=dim
                        )
                    case "collective":
                        mapping = self._map_collective(
                            i_frame, data_keys, meta_keys, dim=dim
                        )
            else:
                mapping = self._map_analog(i_frame, data_keys, dim=dim)
            if mapping:
                loc[i_frame] = mapping
            else:
                return
        # perform indexing across files, and or datasets and metadata
        index: dict[int, list[tuple]] = {}
        if self.collate:
            shift = 0
            for i_frame in loc:
                mapping = loc[i_frame]
                for j in mapping:
                    index[j + shift] = [(i_frame,) + mapping[j]]
                shift += max(loc[i_frame].keys()) + 1
        else:
            for i_frame in loc:
                mapping = loc[i_frame]
                for j in mapping:
                    if j in index:
                        index[j].append((i_frame,) + mapping[j])
                    else:
                        index[j] = [(i_frame,) + mapping[j]]
        self.index = index
        return self.index

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return max(self.index.keys()) + 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Dataset shape.

        Note: If the dataset from different frames have different shape, the one
          with maximal number of elements is chosen.
        """
        if self.collate:
            n_samples = len(self)
            sample = self[0][0]
            samples = [
                self[sum(self.frame_size[:i])][0] for i in range(len(self.frame_size))
            ]
        else:
            n_samples = len(self) * len(self.frame)
            samples = self[0][0]
        dim = 0
        for i, s in enumerate(samples):
            if dim < np.prod(s.shape):
                dim = np.prod(s.shape)
                sample = s
        return (n_samples, *sample.shape)

    @property
    def byte_size(self):
        """Size of a sample (tensor and metadata) in bytes."""
        if self.collate:
            sample, metadata = self[0]
        else:
            sample, metadata = self[0][0][-1], self[0][1][-1]
        sample_bytes = estimate_byte_size(sample)
        metadata_bytes = estimate_byte_size(metadata)
        return sample_bytes, metadata_bytes

    def __getitem__(self, idx: int) -> tuple:
        """Get sample tensors and metadata dictionaries at specified index.

        Returns:
          (torch.Tensor, dict) if collate=True, otherwise
          (tuple[torch.Tensor, ...], tuple[dict, ...])
        """
        # cache lookup
        if self.collate:
            if self.cache[0] is not None and idx in self.cache[0]:
                return self.cache[0][idx], self.cache[1][idx]
        elif self.cache[0] is not None and idx * len(self.frame) in self.cache[0]:
            idcs_across = range(idx * len(self.frame), (idx + 1) * len(self.frame))
            return (
                tuple(self.cache[0][i] for i in idcs_across),
                tuple(self.cache[1][i] for i in idcs_across),
            )

        # create index map if necessary
        if not hasattr(self, "index"):
            self.make_index(dim=self.dim, scheme=self.scheme, collate=self.collate)
        if idx not in self.index:
            raise IndexError(f"{idx} not found in index.")

        # use index map to read samples from frame
        samples: tuple[torch.Tensor, ...] = ()
        metadata: tuple[dict, ...] = ()
        for loc in self.index[idx]:
            # fetch sample(s)
            frame_index, key, attr_key = loc
            sample_arr = self.frame[frame_index][key][idx]
            sample = torch.Tensor(sample_arr)
            if len(sample) == 2 or (
                self.n_channels and sample.shape[0] != self.n_channels
            ):
                sample = sample.view(1, *sample.shape)
            if self.n_channels is None:
                self.n_channels = sample.shape[0]
            samples += (sample,)
            # fetch sample metadata
            info = dict(self.frame[frame_index][attr_key].attrs)
            metadata += (info,)

        # cache samples and metadata
        if self.cache[0] is not None and self.cache[1] is not None:
            if self.collate:
                for i, sample in enumerate(samples):
                    self.cache[0][idx + i] = sample
                    self.cache[1][idx + i] = metadata[i]
            else:
                for i, sample in enumerate(samples):
                    self.cache[0][idx * len(self.frame) + i] = sample
                    self.cache[1][idx * len(self.frame) + i] = metadata[i]

        if len(samples) > 1:
            return samples, metadata
        else:
            return samples[0], metadata[0]

    def purge_cache(self):
        """Purge cache."""
        for c in self.cache:
            if c is not None:
                c.clear_allocation()

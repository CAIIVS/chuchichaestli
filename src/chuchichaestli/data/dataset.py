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
import torch
from torch.utils.data import Dataset
# from chuchichaestli.data.cache import SharedArray
from typing import Any
from collections.abc import Callable


class HDF5Dataset(Dataset):
    """Dataset for loading HDF5 frames."""

    def __init__(
        self,
        path: str | Path,
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
        cache_size: int | float | str | None = "2G",
        **kwargs,
    ):
        """Constructor.

        Args:
          path: Path to a file or data directory containing the HDF5 files
          file_key: Key to filter particular files
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
          cache_size: If not None, a cache is allocated for stochastic caching.
          kwargs: Keyword arguments for h5py.File.
            See https://docs.h5py.org/en/stable/high/file.html#h5py.File
        """
        self.path = Path(path)
        if self.path.is_file() and self.path.suffix == ".hdf5":
            self.files = [self.path]
        else:
            self.files = sorted([f for f in self.path.rglob(file_key) if f.is_file()])
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

        if self.scheme in ["bijective", "collective"]:
            raise NotImplementedError(f"{self.scheme} is not implemented yet.")

        self.load_frame(**self.frame_args)
        self.make_index(dim=self.dim, scheme=self.scheme, collate=self.collate)
        # TODO: add shared cache for data and metadata
        if cache_size is None:
            self.cache = [None, None]
        else:
            self.cache_push(self[0])
        if self.preload:
            pass

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
    def frame_structure(self) -> dict[int, list[str]]:
        """Fetch unfiltered frame structure (HDF5 Groups) of each frame."""
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
                if len(group.shape) == 2:
                    iN = 1
                else:
                    iN = group.shape[dim]
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
        """Index the frames for individual samples.

        The index map has the shape:
          [frame index, group index, dataset index]
            or
          [frame index, group index, attribute key]

        Possible scenarios:
          1 frame, 1 group index, 1 dataset
          1 frame, multiple group indices for multiple datasets

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
        self.length = max(index.keys()) + 1
        return self.index

    def __len__(self):
        """Number of samples in the dataset."""
        return self.length

    def __getitem__(
        self,
        item: int,
    ) -> tuple[torch.Tensor, dict] | tuple[tuple[torch.Tensor, ...], tuple[dict, ...]]:
        """Get samples and metadata at specified index."""
        if not hasattr(self, "index"):
            self.make_index(dim=self.dim, scheme=self.scheme, collate=self.collate)
        if item not in self.index:
            raise IndexError(f"{item} not found in index.")

        samples: tuple[torch.Tensor, ...] = ()
        metadata: tuple[dict, ...] = ()
        for loc in self.index[item]:
            # fetch sample
            frame_index, key, attr_key = loc
            sample_arr = self.frame[frame_index][key][frame_index]
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
        if len(samples) > 1:
            return samples, metadata
        else:
            return samples[0], metadata[0]

    def cache_push(self, index):
        """Fetch an index and cache it."""
        # TODO
        pass



if __name__ == "__main__":
    data_dir = Path(__file__).parents[3] / "data"
    file_key = "240818_tng50-1_dm_50_*"
    ds = HDF5Dataset(data_dir, file_key, meta_groups="**/metadata/*")
    print(ds[0][1])

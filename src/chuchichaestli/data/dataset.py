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
import tempfile
import pprint
import h5py
from h5py import (
    File as H5PyFile,
    Dataset as H5PyDataset,
    AttributeManager as H5PyAttrs,
)
import torch
from torch.utils.data import Dataset
from chuchichaestli.data.cache import (
    nbytes,
    npy_to_torch_dtype,
    serial_byte_size,
    SharedArray,
    SharedDictList,
)
from typing import Any
from collections.abc import Callable
import warnings

H5_FILE_EXTENSIONS = [".hdf", ".h5", ".hdf5", ".he5"]


class HDF5Dataset(Dataset):
    """Dataset for loading HDF5 frames with optional shared memory (sto)caching.

    Constructs a `frame` of single or multiple HDF5 file objects.
    Can be selectively pointed to HDF5 Groups containing HDF5 Datasets via the `groups` keyword.
    If a frame has multiple datasets, the `collate` argument determines whether they are read in
    major-column or major-row ordering across files in a frame.
    A frame can be read in three different modes (always take effect after collate):
    - contiguous (parallel=False; sequentializes the datasets and returns single samples)
    - parallel (parallel=True; multiple samples, one for each dataset)
    - pair (pair=True; read samples pair-wise parallel)
    - custom (collate is callable; use your own ordering by passing a function to collate and/or attrs_retrieval)
    Note that caching only works for datasets with compatible shapes (i.e. image tensors need same dimensions).
    """

    data_ignores = (
        "metadata",
        "attributes",
        "attrs",
        "label",
        "annotations",
        "targets",
        "mask",
    )
    attrs_ignores = ()

    def __init__(
        self,
        path: str | Path | list[str] | list[Path],
        groups: str | tuple[str, ...] = "*",
        sort_key: Callable | None = None,
        collate: bool | Callable = False,
        parallel: bool = False,
        pair: bool = False,
        squash: bool = True,
        dtype: torch.dtype = torch.float32,
        attr_groups: str | tuple[str, ...] | None = None,
        attrs_retrieval: str | Callable = "auto",
        return_as: str | dict | None = "tuple",
        preload: bool = False,
        cache: int | float | str | bool | None = "4G",
        attrs_cache: int | float | str | bool | None = "64M",
        **kwargs,
    ):
        """Constructor.

        Args:
            path: Path to a file, files, or data directory which contains the HDF5 files. May
              contain wildcards `*` and `**`.
            groups: Location(s) or filter pattern(s) for HDF5 groups containing HDF5 datasets
              (allows wildcards `*` and `**`). Note, if wildcard expressions have too many
              matches, the file tree search could take a long time, especially when there are
              many HDF5Groups in the file. Thus, it is recommended to not use too generic
              wildcard expressions. Alternatively, use `data_ignores` to tune the file tree search.
            sort_key: Sorting key function for the frame tree of HDF5 groups.
            collate: If True, multiple HDF5 files are collated into a single dataset. This is
              equivalent to reading files in Fortran-like row-major ordering. If False,
              __getitem__ tries to pull samples from each file. Can also be a function that
              creates a custom ordering with signature
              `func(hdf5_datasets: list[list]) -> list[list]`.
            parallel: If True, HDF5 datasets are read in parallel, resulting in multiple samples
              being pulled, otherwise datasets are concatenated and sequentialized (note that
              contiguous reading (with squash=True) only works, if dataset dimensions are
              compatible).
            pair: If True, datasets are paired and read pairwise-parallel
              (automatically sets parallel=True).
            squash: Squash dimension along first axis via summation (default behaviour
              for sequential, i.e. parallel=False reads).
            dtype: Data tensor type; default: torch.float32.
            attr_groups: Locations or filter pattern(s) for HDF5 groups containing HDF5 attributes
              (allows wildcards '*' and '**'). If empty or None, no attributes will be filtered.
              Note: If wildcard expressions have too many matches, the file tree search could take a
              long time, especially when there are many HDF5Groups in the file. Thus, it is
              recommended to not use too generic wildcard expressions. Alternatively,
              use `attrs_ignores` to tune the file tree search.
            attrs_retrieval: A retrieval strategy for attributes; can be a function
              one of ['auto' | None | Callable].
            return_as: Return type of the dataset; one of ['tuple', 'dict', dict, None].
              If a dictionary object is give, then the keys in the dictionary are used to
              map items onto it.
            preload: Preload and cache the dataset.
            cache: If not None or False, memory is allocated for stochastic caching of the tensors.
            attrs_cache: If not None or False, memory is allocated for stochastic caching
              of ancillary data.
            kwargs: Keyword arguments for `h5py.File`. See
              `https://docs.h5py.org/en/stable/high/file.html#h5py.File`.
        """
        self.frame_args = kwargs
        self.frame_args.setdefault("libver", "latest")
        self.collate = collate
        if callable(collate):
            self.collate = True
            self._collate_fn = staticmethod(collate)
        else:
            self._collate_fn = self._collate
        self.pair = pair
        self.parallel = parallel or self.pair
        self.squash = squash
        self.dtype = dtype
        self.return_as = return_as
        self.preload = preload

        # build frame
        self._frame: list[H5PyFile]
        self._virt_frame: list[H5PyFile]
        self._frame_tree: list[list[str]]
        self.files = self.glob_path(path)
        self.load_frame(**self.frame_args)

        # locate HDF5 datasets in frame
        self.frame_datasets: list[list[H5PyDataset]] = []
        self.groups = (groups,) if not isinstance(groups, list | tuple) else groups
        self.sort_key = sort_key if sort_key is not None else self.default_sort_key
        self.pin_data(self.groups)
        self.make_index()

        # ancillary data from HDF5 group attributes (or datasets) in frame
        self.frame_attrs: list[list[H5PyAttrs | H5PyDataset]] = []
        self.attr_groups_selected = attr_groups
        self.attrs_retrieval = attrs_retrieval
        if not isinstance(attr_groups, list | tuple) and attr_groups is not None:
            self.attr_groups_selected = (attr_groups,)
        self.pin_attrs(self.attr_groups_selected)

        # allocate memory for cache
        self.cache: tuple[list[SharedArray] | None, list[SharedDictList] | None]
        if isinstance(cache, bool) and cache:
            cache = (
                self.serial_size[0]
                if self.serial_size[0] < nbytes("4G")
                else nbytes("4G")
            )
        if isinstance(attrs_cache, bool):
            attrs_cache = self.serial_size[1]
        self.init_cache(cache, attrs_cache)
        if self.preload and self.cache[0] is not None:
            for i in range(len(self)):
                self[i]

    @staticmethod
    def _split_glob(path: str | list[str]) -> tuple[list[str], list[str | None]]:
        """Split path containing wildcard into root and wildcard expression."""
        root: list[str]
        file_key: list[str | None]
        if isinstance(path, str) and "*" in path:
            path = path[1:] if path.startswith("/") else path
            components = Path(path).parts
            iwc = [i for i, c in enumerate(components) if "*" in c][0]
            root = [str(Path().joinpath(*components[:iwc]))]
            file_key = [str(Path().joinpath(*components[iwc:]))]
        elif isinstance(path, list | tuple):
            root = []
            file_key = []
            for p in path:
                f, k = HDF5Dataset._split_glob(p)
                root += f
                file_key += k
        else:
            root = [str(path)]
            file_key = [None]
        return root, file_key

    @staticmethod
    def glob_path(path: str | Path | list[str] | list[Path]) -> list[Path]:
        """Glob path recursively for HDF5 files.

        Args:
            path: Filename, path or list, can contain wildcards `*` or `**`.
        """
        files: list[Path] = []
        root, file_key = HDF5Dataset._split_glob(path)
        for p, k in zip(root, file_key):
            if k is None:
                files.append(Path(p))
            else:
                path_files = [
                    f
                    for f in Path(p).rglob(k)
                    if f.is_file() and f.suffix in H5_FILE_EXTENSIONS
                ]
                files += sorted(path_files)
        files = [p for p in files if p.exists()]
        return files

    def __str__(self):
        """Instance string."""
        instance = f"{self.__class__.__name__}"
        instance += "{"
        instance += f"#f{self.n_files}d{self.n_datasets},"
        instance += f"g:[{','.join(self.groups)}]"
        instance += f"{',squash' if self.squash else ''}"
        instance += f"{',collate' if self.collate else ''}"
        instance += f"{',pair' if self.pair else ''}"
        instance += f"{',parallel' if self.parallel and not self.pair else ''}"
        instance += "}"
        return instance

    def info(
        self,
        print_: bool = True,
        show_data_info: bool = True,
        show_attrs_info: bool = True,
    ) -> str:
        """Print summary information about the frame.

        Args:
            print_: Print to stdout, otherwise only return summary string.
            show_data_info: Include detailed info about the datasets in the frame.
            show_attrs_info: Include detailed info about the attribute sets in the frame.
        """
        dims = self.dims if isinstance(self.dims, tuple) else self.indexed_dims
        summary = str(self)
        summary += "\n" + "-" * 50 + "\n"
        summary += f"Serial size:             \t ({self.serial_size[0].as_str()}, {self.serial_size[1].as_str()})\n"
        summary += f"Sample size:             \t ({self.sample_serial_size[0].as_str()}, {self.sample_serial_size[1].as_str()})\n"
        summary += f"Cache size:              \t {self.cached_bytes.as_str()} / {self.cache_size.as_str()}\n"
        summary += f"Number of files:         \t {self.n_files}\n"
        summary += f"Selected groups:         \t {self.groups}\n"
        summary += f"Number of datasets:      \t {self.n_datasets}\n"
        summary += f"Number of samples:       \t {len(self)}\n"
        summary += f"Selected attrs:          \t {self.attr_groups_selected}\n"
        summary += f"Number of attribute sets:\t {self.n_attrsets}\n"
        summary += f"Number of attrs:         \t {self.n_attrs}\n"
        if show_data_info:
            data_info = [
                [f"H5PyDataset={di}, shape={si}" for di, si in zip(d, s)]
                for d, s in zip(self.data_groups, self.raw_dims)
            ]
            summary += f"Data {dims}:\n{pprint.pformat(data_info)}\n"
        if show_attrs_info:
            attrs_info = self.frame_attrs
            summary += f"Attrs:\n{pprint.pformat(attrs_info)}\n"
        if print_:
            print(summary)
        return summary

    def load_frame(self, **kwargs) -> list[H5PyFile] | None:
        """Load HDF5 file instances.

        Args:
            kwargs: Keyword arguments for `h5py.File`.
        """
        self._frame = [H5PyFile(f, "r", **kwargs) for f in self.files]
        self._virt_frame = []
        return self._frame

    @property
    def frame(self) -> list[H5PyFile]:
        """Lazy-loading list of HDF5 file instances."""
        if not hasattr(self, "_frame"):
            self.load_frame(**self.frame_args)
        return self._frame

    def load_frame_tree(self, sort_key: Callable | None = None) -> list[list[str]]:
        """Load tree of HDF5 group for each frame (HDF5 file instance).

        Args:
            sort_key: Sorting key function for the list of HDF5 groups.

        Note:
            This method walks through the entire HDF5 group tree and could take some time.
        """
        tree: list[list[str]] = []
        sort_key = sort_key if sort_key else self.sort_key
        for i, f in enumerate(self.frame):
            tree.append([])
            f.visit(tree[i].append)
        self._frame_tree = [sorted(t, key=sort_key) for t in tree]
        return self._frame_tree

    @staticmethod
    def default_sort_key(x: Any) -> int:
        """Default sort key for HDF5 groups in the frame tree."""
        return int(x.split("/")[-1]) if x.split("/")[-1].isdigit() else 0

    @property
    def frame_tree(self) -> list[list[str]]:
        """Fetch unfiltered frame structure (HDF5 Groups) of each frame (HDF5 File instance)."""
        if not hasattr(self, "_frame_tree"):
            self.load_frame_tree()
        return self._frame_tree

    def filter_tree(
        self,
        groups: tuple[str, ...],
        omit_keys: tuple[str, ...] | None = None,
        strip_root: bool = True,
    ) -> list[list[str]]:
        """Filter frame structure (HDF5 Groups) of each frame (HDF5 File instance).

        Args:
            groups: Names of groups to check against the entire frame structure.
            omit_keys: Keywords to ignore while filtering
              (for speed-up when `frame_tree` is massive).
            strip_root: If True, the groups starting with `/` are stripped.
        """
        filtered_groups: tuple[str, ...] = ()
        for g in groups:
            g = g[1:] if g.startswith("/") else g
            for i, t in enumerate(self.frame_tree):
                matches = fnmatch.filter(t, g)
                if omit_keys:
                    matches = [m for m in matches if not any(k in m for k in omit_keys)]
                filtered_groups += tuple(matches)
        return tuple(dict.fromkeys(filtered_groups).keys())

    def pin_data(self, groups: tuple[str, ...] | None = ("*",)) -> list[H5PyDataset]:
        """Find and pin datasets in the frame.

        Args:
            groups: Location(s) or filter pattern(s) for HDF5 groups containing
              HDF5 datasets (allows wildcards '*' and '**'). Note, if `None`, the
              entire HDF5 tree will be checked, which could take a long time for
              HDF5 files that contain many groups.
        """
        datasets: list[list[H5PyDataset]] = []
        if groups is None:
            groups = self.groups or ("*",)
        if any("*" in g for g in groups):
            groups = self.filter_tree(groups, omit_keys=self.data_ignores)
        for f in self.frame:
            datasets.append(
                [f[g] for g in groups if g in f and isinstance(f[g], H5PyDataset)]
            )
        self.frame_datasets = datasets
        return self.frame_datasets

    def pin_attrs(
        self, attr_groups: tuple[str, ...] | None = None
    ) -> list[H5PyAttrs | H5PyDataset]:
        """Find and pin attribute sets (metadata) in the frame.

        Args:
            attr_groups: Location(s) or filter pattern(s) for HDF5 groups
              containing HDF5 attributes (allows wildcards '*' and '**'). Note,
              if `None`, no attributes are pinned.

        Note, if no attribute sets are found, separate (other than group-selected)
        datasets are pinned.
        """
        attrs: list[list[H5PyAttrs | H5PyDataset]] = []
        if attr_groups is None or not attr_groups:
            self.frame_attrs = []
        if attr_groups and any("*" in g for g in attr_groups):
            attr_groups = self.filter_tree(attr_groups)
        if attr_groups:
            for i, f in enumerate(self.frame):
                # look for HDF5 Attributes
                attrs_list = [
                    f[g].attrs for g in attr_groups if g in f and f[g].attrs.keys()
                ]
                # if no HDF5 Attributes found, look for other HDF5 Datasets
                if not attrs_list:
                    attrs_list = [
                        f[g]
                        for g in attr_groups
                        if g in f and isinstance(f[g], H5PyDataset)
                    ]
                    # check if attrs are just data and filter the list
                    if len(self.frame_datasets) > i:
                        data_list = self.frame_datasets[i]
                        for d in data_list:
                            for i, a in enumerate(attrs_list):
                                attr_is_data = (
                                    d.shape == a.shape
                                    and d.size == a.size
                                    and d.nbytes == a.nbytes
                                    and d.name == a.name
                                )
                                if attr_is_data:
                                    attrs_list.pop(i)
                if attrs_list:
                    attrs.append(attrs_list)
        self.frame_attrs = attrs
        return self.frame_attrs

    @property
    def data_groups(self):
        """HDF5 groups of the selected datasets in the frame."""
        if self.frame_datasets:
            return [[d.name for d in dl] for dl in self.frame_datasets]
        return self.groups

    @property
    def attr_groups(self):
        """HDF5 groups of the selected attribute sets in the frame."""
        if self.frame_attrs:
            attr_groups = [
                [attr.name for attr in attr_list if isinstance(attr, H5PyDataset)]
                for attr_list in self.frame_attrs
            ]
            if any(attr_groups):
                return attr_groups
        return self.attr_groups_selected

    @property
    def n_files(self):
        """Number of files in the frame."""
        return len(self.frame)

    @property
    def n_datasets(self):
        """Number of datasets in the frame."""
        return sum([len(dl) for dl in self.frame_datasets])

    @property
    def n_samples(self):
        """Number of samples in the frame."""
        return len(self)

    @property
    def cached_items(self):
        """Number of cached samples from the frame."""
        if hasattr(self, "cache") and self.cache[0] is not None:
            return sum([c.cached_states for c in self.cache[0]])
        return 0

    @property
    def n_attrsets(self):
        """Number of attribute sets in the frame."""
        return sum([len(attrs) for attrs in self.frame_attrs])

    @property
    def n_attrs(self):
        """Number of attribute sets in the frame."""
        if self.frame_attrs and isinstance(self.frame_attrs[0][0], H5PyDataset):
            return sum([sum(len(a) for a in attrs) for attrs in self.frame_attrs])
        else:
            return sum([len(attrs) for attrs in self.frame_attrs])

    @property
    def cached_attrs(self):
        """Number of cached samples from the frame."""
        if hasattr(self, "cache") and self.cache[1] is not None:
            return sum([c.cached_states for c in self.cache[1]])
        return 0

    def _stitch_datasets(
        self,
        datasets: list[H5PyDataset],
        layout: tuple | list[tuple] | None = None,
        virt_label: str = "data",
    ) -> H5PyDataset:
        """Concatenate datasets via HDF5 Virtual Datasets feature."""
        if len(datasets) == 1:
            return datasets[0]
        replicate = 1
        shapes = [d.shape for d in datasets]
        lengths = [sh[0] for sh in shapes]
        sections = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
        if layout is None:
            dims = tuple(set([sh[1:] for sh in shapes]))[0]
            layout = (sum(lengths),) + dims
        elif isinstance(layout, list):
            layout = (sum(sh[0] for sh in layout),) + layout[0][1:]
        if layout[0] > sum(lengths) and layout[0] % sum(lengths) == 0:
            replicate = layout[0] // sum(lengths)
            sections = [replicate * s for s in sections]
        vdl = h5py.VirtualLayout(shape=layout, dtype=datasets[0].dtype)
        uid = "_".join([str(ds.id.id)[:8] for ds in datasets])
        tmpdir = tempfile.mkdtemp()
        virt_fname = Path(tmpdir).joinpath(f"VDS_{uid}.h5")
        with h5py.File(virt_fname, "w", libver="latest") as f:
            for left, right, ds in zip(sections, sections[1:], datasets):
                delta = (right - left) // replicate
                for i in range(left, right, delta):
                    vdl[i : i + delta] = h5py.VirtualSource(ds)
            f.create_virtual_dataset(virt_label, vdl)
        self._virt_frame.append(h5py.File(virt_fname, "r", libver="latest"))
        stitched = self._virt_frame[-1][virt_label]
        return stitched

    def _stitch_attrs(
        self,
        attrs: list[H5PyAttrs | H5PyDataset],
        data_layout: tuple | list[tuple] | None = None,
    ):
        """Concatenate attributes via HDF5 Virtual Datasets feature."""
        if len(attrs) == 1 and len(attrs[0]) == data_layout[0]:
            return attrs[0]
        elif self.attrs_is_data:
            if isinstance(data_layout, tuple):
                sample_dims = tuple(set([a.shape[1:] for a in attrs]))[0]
                layout = data_layout[:1] + sample_dims
                stitched_attrs = self._stitch_datasets(
                    attrs, layout=layout, virt_label="attrs"
                )
            else:
                sample_dims = [a.shape[1:] for a in attrs]
                layout = [dl[:1] + sd for dl, sd in zip(data_layout, sample_dims)]
                stitched_attrs = self._stitch_datasets(
                    attrs, layout=layout, virt_label="attrs"
                )
            return stitched_attrs
        elif self.attrs_is_dict:
            if len(attrs) == data_layout[0]:
                return attrs
            return attrs * data_layout[0]

    @staticmethod
    def _collate(
        data: list[list],
        fn: Callable = lambda x: x,
        pad: bool = False,
        pad_val: Any = None,
    ) -> list[list]:
        """Collate list of lists, i.e. reorder data as column-major to row-major.

        Args:
            data: Data to be reordered.
            fn: Function or iterator to reorder data before collation.
            pad: If True, reordered lists will be padded (useful if lists need to be zipped).
            pad_val: Value to use for list padding (default: None).
        """
        pad_len = max(len(d) for d in data)
        padded_data = [d + [pad_val] * (pad_len - len(d)) for d in fn(data)]
        if pad:
            collated = [[di for di in d] for d in zip(*padded_data)]
        else:
            collated = [
                [di for di in d if di is not pad_val] for d in zip(*padded_data)
            ]
            collated = [d for d in collated if len(d)]
        return collated

    @staticmethod
    def _contiguate(
        data: list[list],
        pad: bool = True,
        pad_val: Any = None,
    ) -> list[list]:
        """Contiguate/Flatten list of lists.

        Args:
            data: Data to be reordered.
            pad: If True, reordered lists will be padded (useful if lists need to be zipped).
            pad_val: Value to use for list padding (default: None).
        """
        if pad:
            contiguous = [[di for d in data for di in d]]
        else:
            contiguous = [[di for d in data for di in d if di is not pad_val]]
        return contiguous

    @staticmethod
    def _parallelize(
        data: list[list],
        pad: bool = False,
        pad_val: Any = None,
    ) -> list[list]:
        """Parallelize list of lists, i.e. expand (last) dimension if trivial.

        Args:
            data: Data to be reordered.
            pad: If True, reordered lists will be padded (useful if lists need to be zipped).
            pad_val: Value to use for list padding (default: None).
        """
        parallelized = data
        if len(data) == 1 and len(data[0]) >= 1:
            if not pad:
                parallelized = [[di] for d in data for di in d if di is not pad_val]
            else:
                parallelized = [[di] for d in data for di in d]
        return parallelized

    @staticmethod
    def _pair(
        data: list[list],
        pad: bool = True,
        pad_val: Any = None,
    ) -> list[list]:
        """Pair list of lists, i.e. reorder data into two separate lists.

        Args:
            data: Data to be reordered.
            pad: If True, reordered lists will be padded (useful if lists need to be zipped).
            pad_val: Value to use for list padding (default: None).
        """
        if len(data) % 2:
            warnings.warn("Trying to pair odd number of data subsets.")
        if pad:
            paired = [
                [di for d in data[: len(data) // 2] for di in d],
                [di for d in data[len(data) // 2 :] for di in d],
            ]
        else:
            paired = [
                [di for d in data[: len(data) // 2] for di in d if d is not pad_val],
                [di for d in data[len(data) // 2 :] for di in d if d is not pad_val],
            ]
        return paired

    @staticmethod
    def is_aligned(dims: tuple | list[list[tuple]]) -> bool:
        """Test if the model is aligned, i.e. same sample size for parallel reads.

        Args:
            dims: Dimensions to be tested for alignment.
        """
        if isinstance(dims, tuple):
            return True
        sizes = [[di[0] for di in d] for d in dims]
        collated_sizes = HDF5Dataset._collate(sizes)
        return all([len(set(cs)) == 1 for cs in collated_sizes])

    def make_index(self) -> list[H5PyDataset]:
        """Index frame datasets, apply collation and parallelization settings."""
        if not hasattr(self, "indexed_datasets"):
            dims = self.dims
            datasets = self.frame_datasets
            if not datasets:
                self.indexed_datasets = datasets
                return None
            if self.collate:
                datasets = self._collate_fn(datasets)
            if self.pair:
                datasets = self._pair(datasets)
            if self.parallel or self.pair:
                datasets = self._parallelize(datasets)
            elif self.squash and isinstance(dims, tuple):
                datasets = self._contiguate(datasets)
            dims = [dims] if isinstance(dims, tuple) else dims
            datasets = [
                self._stitch_datasets(ds, layout=sh) for ds, sh in zip(datasets, dims)
            ]
            self.indexed_datasets = datasets
        return self.indexed_datasets

    @property
    def raw_dims(self) -> list[list[tuple]]:
        """Raw dataset dimensions (ignores collation and parallelization settings)."""
        return [[d.shape for d in dl] for dl in self.frame_datasets]

    @property
    def dims(self) -> tuple | list[list[tuple]]:
        """Dataset dimensions (respecting collation and parallelization settings)."""
        dims = self.raw_dims
        if not dims:
            return dims
        if self.collate:
            dims = self._collate_fn(dims)
        if self.pair:
            dims = self._pair(dims)
        if self.parallel:
            dims = self._parallelize(dims)
        elif self.squash:
            dims = self._contiguate(dims)
            sample_dims = set([d[1:] for d in dims[0]])
            if len(sample_dims) > 1:
                raise ValueError(
                    "Sample dimensions differ across HDF5 datasets! "
                    "Try setting `squash = False` or selecting appropriate HDF5 groups."
                )
            squashed_dims = (sum([d[0] for d in dims[0]]),) + tuple(sample_dims)[0]
            return squashed_dims
        if len(dims) > 1 and not self.is_aligned(dims):
            warnings.warn(
                f"Dimensions of datasets {dims} are not aligned which may lead to unintended behaviour!"
            )
        return dims

    @property
    def indexed_dims(self) -> list[tuple]:
        """Dataset dimension of the indexed datasets."""
        if hasattr(self, "indexed_datasets"):
            return [ds.shape for ds in self.indexed_datasets]
        return self.dims

    def __len__(self) -> int:
        """Dataset length."""
        dims = self.dims
        if len(dims) == 0:
            return 0
        elif isinstance(dims, tuple):
            return dims[0]
        sizes = [[di[0] for di in d] for d in dims]
        if self.parallel:
            return min([sum(s) for s in sizes])
        else:
            return sum([sum(s) for s in sizes])

    def get_cached_item(self, index: int) -> list[torch.Tensor | None]:
        """Fetch item(s) if in cache."""
        if hasattr(self, "cache") and self.cache[0] is not None:
            return [c[index] for c in self.cache[0]]
        return [None for _ in self.indexed_dims]

    def cache_item(
        self,
        index: int,
        items: torch.Tensor | list[torch.Tensor],
        output_index: int | None = None,
        overwrite: bool = False,
    ):
        """Cache item (overwrite by default off).

        Args:
            index: Index where to cache the attribute.
            items: Item(s) to be cached.
            output_index: Index of the list of caches, if output contains multiple items.
            overwrite: If True and there already is a cached element at the given index,
              that item is overwritten.
        """
        if not hasattr(self, "cache") or self.cache[0] is None:
            return
        if output_index is None:
            output_index = 0
        if isinstance(items, torch.Tensor):
            previous = self.cache[0][output_index][index]
            if previous is None or overwrite:
                self.cache[0][output_index][index] = items
        else:
            for i, t in enumerate(items):
                previous = self.cache[0][i][index]
                if previous is None or overwrite:
                    self.cache[0][i][index] = t

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Get sample tensors at specified index."""
        length = len(self)
        if index >= length:
            raise IndexError(f"Index {index} exceeds the dataset length.")
        if index < 0:
            index = index % length
        # look for cached items
        items = self.get_cached_item(index)
        non_cached = [i for i, item in enumerate(items) if item is None]
        # if not found in cache, read item(s) from file
        if non_cached:
            datasets = self.make_index()
            for i in non_cached:
                item = torch.from_numpy(datasets[i][index]).type(self.dtype)
                items[i] = item
                self.cache_item(index, item, i)
        attrs = self.get_cached_attrs(index)
        if attrs is None or self.n_attrs:
            attrs = self.attrs_getter(index)
            for i, attr in enumerate(attrs):
                self.cache_attrs(index, attr, i)
        if len(items) == 1:
            if not attrs:
                return items[0]
            else:
                return items[0], attrs[0] if len(attrs) == 1 else tuple(
                    a for a in attrs
                )
        elif self.return_as == "tuple":
            if not attrs:
                return tuple(t for t in items)
            else:
                return tuple(t for t in items), tuple(a for a in attrs)
        elif self.return_as == "dict":
            if not attrs:
                return {k: v for k, v in enumerate(items)}
            else:
                return {k: (v, a) for k, (v, a) in enumerate(zip(items, attrs))}
        elif isinstance(self.return_as, dict):
            if not attrs:
                return {k: v for k, v in zip(self.return_as.keys(), items)}
            else:
                return {
                    k: (v, a) for k, v, a in zip(self.return_as.keys(), items, attrs)
                }
        elif not attrs:
            return items
        else:
            return items, attrs

    @property
    def attrs_is_dict(self) -> bool:
        """If True, attribute sets are represented by dict or h5py.AttributeManager objects."""
        if self.frame_attrs and self.frame_attrs[0]:
            # is_data = all([isinstance(a, dict | H5PyAttrs) for a in self.frame_attrs[0]])
            is_data = isinstance(self.frame_attrs[0][0], dict | H5PyAttrs)
            return is_data
        return False

    @property
    def attrs_is_data(self) -> bool:
        """If True, attribute sets are represented by h5py.Dataset objects."""
        if self.frame_attrs and self.frame_attrs[0]:
            # is_data = all([isinstance(a, H5PyDataset) for a in self.frame_attrs[0]])
            is_data = isinstance(self.frame_attrs[0][0], H5PyDataset)
            return is_data
        return False

    @property
    def attrs_dims(self) -> list[tuple] | None:
        """Output format of attributes."""
        if not hasattr(self, "indexed_attrs"):
            self.attrs_getter(0)
        if hasattr(self, "indexed_attrs"):
            if self.attrs_is_data:
                dims = [a.shape for a in self.indexed_attrs]
            elif self.attrs_is_dict:
                dims = [(len(a),) for a in self.indexed_attrs]
            return dims

    @property
    def attrs_dtype(self) -> list[torch.dtype]:
        """Data type of attributes if represented as HDF5 Dataset."""
        if not hasattr(self, "indexed_attrs"):
            self.attrs_getter(0)
        if hasattr(self, "indexed_attrs"):
            if self.attrs_is_data:
                return [npy_to_torch_dtype(a.dtype) for a in self.indexed_attrs]

    def get_cached_attrs(self, index: int) -> list[torch.Tensor | dict | None]:
        """Fetch attrs if in cache."""
        if hasattr(self, "cache") and self.cache[1] is not None:
            return [c[index] for c in self.cache[1]]
        return None

    def cache_attrs(
        self,
        index: int,
        attrs: torch.Tensor | dict | list[torch.Tensor | dict],
        output_index: int | None = None,
        overwrite: bool = False,
    ):
        """Cache attrs (overwrite by default off).

        Args:
            index: Index where to cache the attribute.
            attrs: Attribute(s) to be cached.
            output_index: Index of the list of caches, if output contains multiple attributes.
            overwrite: If True and there already is a cached element at the given index,
              that attribute is overwritten.
        """
        if not hasattr(self, "cache") or self.cache[1] is None:
            return
        if output_index is None:
            output_index = 0
        if isinstance(attrs, torch.Tensor | dict):
            previous = self.cache[1][output_index][index]
            if previous is None or overwrite:
                self.cache[1][output_index][index] = attrs
        else:
            for i, t in enumerate(attrs):
                previous = self.cache[1][i][index]
                if previous is None or overwrite:
                    self.cache[1][i][index] = t

    def attrs_getter(self, index: int) -> list[dict | torch.Tensor]:
        """Retrieve ancillary data item at `index`."""
        if self.n_attrs < 1:
            return []
        if self.attrs_retrieval == "auto":
            if self.n_datasets == self.n_attrsets:
                # data <-> attribute set mapping: 1-to-1
                indexed_attrs = self._make_attrs_index(self.frame_attrs)
            elif (
                self.n_datasets < self.n_attrsets and self.n_attrsets == self.n_samples
            ):
                # data <-> attribute set mapping: 1-to-all
                indexed_attrs = self._make_attrs_index(self.frame_attrs)
            elif self.n_datasets < self.n_attrsets:
                # data <-> attributes mapping: 1-to-many
                indexed_attrs = self._make_many_attrs_index(self.frame_attrs)
            elif self.n_datasets > self.n_attrsets and self.n_attrsets == 1:
                # data <-> attribute set mapping: all-to-1
                if not hasattr(self, "indexed_attrs"):
                    self.indexed_attrs = [self.frame_attrs[0] * self.n_samples]
                indexed_attrs = self.indexed_attrs
            else:
                # data <-> attribute set mapping: many-to-1
                warnings.warn(
                    "Automatic attribute retrieval not possible!\n"
                    f"Found {self.n_datasets} datasets with {self.n_samples} samples in total,\n"
                    "but only {self.n_attrsets} attribute sets.\n"
                    "Try passing a attribute retrieval function\n"
                    "   fn(index, frame_attrs) -> list[H5PyAttrs | H5PyDataset]\n"
                    "that selects the appropriate attributes for a given index."
                )
                indexed_attrs = []
            if self.attrs_is_data:
                return [
                    torch.from_numpy(a[index]).squeeze() for a in indexed_attrs if a
                ]
            else:
                return [dict(a[index]) for a in indexed_attrs if a]
        elif callable(self.attrs_retrieval):
            return self.attrs_retrieval(index, self.frame_attrs)

    def _make_attrs_index(
        self, attrs: list[list[H5PyAttrs | H5PyDataset]], cache: bool = True
    ):
        """Indexed attribute set getter.

        Assumes a 1-to-1 index match between data and attribute items.

        Args:
            attrs: Attribute sets from which to fetch the item.
            cache: Cache the indexed list of attributes in the `indexed_attrs` instance variable.
        """
        if not hasattr(self, "indexed_attrs"):
            dims = self.dims
            if not attrs:
                self.indexed_attrs = attrs
                return None
            if self.collate:
                attrs = self._collate_fn(attrs)
            if self.pair:
                attrs = self._pair(attrs)
            if self.parallel:
                attrs = self._parallelize(attrs)
            elif self.squash and isinstance(dims, tuple):
                attrs = self._contiguate(attrs)
            dims = [dims] if isinstance(dims, tuple) else dims
            attrs = [
                self._stitch_attrs(a, data_layout=sh) for a, sh in zip(attrs, dims)
            ]
            if cache:
                self.indexed_attrs = attrs
            return attrs
        return self.indexed_attrs

    def _make_many_attrs_index(
        self, attrs: list[list[H5PyAttrs | H5PyDataset]], cache: bool = True
    ):
        """Indexed attribute set getter.

        Assumes a 1-to-many index match between data and attribute items.

        Args:
            attrs: Attribute sets from which to fetch the item.
            cache: Cache the indexed list of attributes in the `indexed_attrs` instance variable.
        """
        if not hasattr(self, "indexed_attrs"):
            dims = self.dims
            if not attrs:
                self.indexed_attrs = attrs
                return None
            if self.collate:
                attrs = self._collate_fn(attrs)
            if self.pair:
                attrs = self._pair(attrs)
            if self.parallel:
                attrs = self._parallelize(attrs)
            elif self.squash and isinstance(dims, tuple):
                attrs = self._contiguate(attrs)
            # assume a 1-to-1 match including an additional global attribute set (first element)
            if self.n_attrsets == self.n_samples + 1:
                attrs = [a[i:j] for a in attrs for i, j in zip([0, 1], [1, len(a)])]
            # assume a 1-to-n match and split each attribute set into n groups
            elif self.n_attrsets % self.n_datasets == 0:
                n_attrs_sections = (
                    self.n_attrsets // self.n_datasets if not self.pair else 1
                )
                if self.pair:
                    n_attrs_sections = 1
                attrs_lengths = [len(a) for a in attrs]
                chunk_sizes = [length // n_attrs_sections for length in attrs_lengths]
                attrs = [
                    a[i : i + chunk]
                    for chunk, a in zip(chunk_sizes, attrs)
                    for i in range(0, len(a), chunk)
                ]
            # any other auto cases for 1-n?
            dims = [dims] if isinstance(dims, tuple) else dims
            dims *= len(attrs) // len(dims)
            if self.pair:
                dims = self.indexed_dims
            attrs = [
                self._stitch_attrs(a, data_layout=sh) for a, sh in zip(attrs, dims)
            ]
            if cache:
                self.indexed_attrs = attrs
            return attrs
        return self.indexed_attrs

    @property
    def sample_serial_size(self) -> tuple["nbytes", "nbytes"]:
        """Size of a sample (tensor and metadata) in bytes (when serialized)."""
        if not hasattr(self, "_sample_serial_size"):
            indices = [
                0,
                1,
                len(self) // 4,
                len(self) // 3,
                len(self) // 2,
                2 * len(self) // 3,
                3 * len(self) // 4,
                len(self) - 1,
            ]
            datasets = self.make_index()
            if self.n_attrs > 0:
                samples = [
                    tuple(torch.from_numpy(d[i]).type(self.dtype) for d in datasets)
                    for i in indices
                ]
                metadata = [self.attrs_getter(i) for i in indices]
                # samples, metadata = [i[0] for i in items], [i[1] for i in items]
            else:
                samples = [
                    tuple(torch.from_numpy(d[i]).type(self.dtype) for d in datasets)
                    for i in indices
                ]
                metadata = None
                # samples, metadata = items, None
            sample_bytes = max([int(serial_byte_size(s)) for s in samples])
            metadata_bytes = (
                max([int(serial_byte_size(m)) for m in metadata]) if metadata else 0
            )
            self._sample_serial_size = nbytes(sample_bytes), nbytes(metadata_bytes)
        return self._sample_serial_size

    @property
    def serial_size(self) -> tuple["nbytes", "nbytes"]:
        """Size of the entire dataset in bytes (when serialized)."""
        if not hasattr(self, "_serial_size"):
            self._serial_size = (
                self.n_samples * self.sample_serial_size[0],
                self.n_attrs * self.sample_serial_size[1],
            )
        return self._serial_size

    @property
    def cached_bytes(self) -> "nbytes":
        """Size of used cache."""
        if self.cache[0] is not None:
            cached_bytes = sum([c.cached_bytes for c in self.cache[0]])
            if self.cache[1] is not None:
                cached_bytes += sum([c.cached_bytes for c in self.cache[1]])
            return cached_bytes
        return nbytes(0)

    @property
    def cache_size(self) -> "nbytes":
        """Size of reserved cache."""
        if self.cache[0] is not None:
            cache_size = sum([c.cache_size for c in self.cache[0]])
            if self.cache[1] is not None:
                cache_size += sum([c.cache_size for c in self.cache[1]])
            return cache_size
        return nbytes(0)

    def init_cache(
        self,
        cache: int | float | str | None = "4G",
        attrs_cache: int | float | str | None = "64M",
    ):
        """Initialize the cache.

        Args:
            cache: Cache size for data items.
            attrs_cache: Cache size for attributes.
        """
        if cache and cache is not None:
            output_length = max(len(self.indexed_dims), 1)
            cache = nbytes(cache) / output_length
            self.cache = (
                [
                    SharedArray(shape=dims, size=cache.as_bstr(), dtype=self.dtype)
                    for dims in self.indexed_dims
                ],
            )
        else:
            self.cache = (None,)
        if self.n_attrs > 0 and attrs_cache is not None:
            output_length = max(len(self.indexed_dims), 1)
            attrs_cache = nbytes(attrs_cache) / output_length
            if self.attrs_is_data:
                self.cache += (
                    [
                        SharedArray(shape=dims, size=attrs_cache.as_bstr(), dtype=dtype)
                        for dims, dtype in zip(self.attrs_dims, self.attrs_dtype)
                    ],
                )
            elif self.attrs_is_dict:
                self.cache += (
                    [
                        SharedDictList(
                            n=dims[0],
                            size=attrs_cache.as_bstr(),
                            slot_size=self.sample_serial_size[1].as_bstr(),
                            descr=f"shm_list_{i}",
                        )
                        for i, dims in enumerate(self.attrs_dims)
                    ],
                )
        else:
            self.cache += (None,)

    def purge_cache(self, reset: bool = False):
        """Purge the cache.

        Args:
            reset: If True, the cache instance attribute is reinitialized.
        """
        cache_sizes = [None, None]
        for i, c in enumerate(self.cache):
            if c is not None:
                cache_sizes[i] = 0
                for ci in c:
                    cache_sizes[i] += ci.cache_size
                    ci.clear_allocation()
        del self.cache
        if reset:
            self.init_cache(*cache_sizes)

    def close(self):
        """Exit the frame, i.e. close all open files and purge the cache."""
        self.purge_cache(reset=False)
        for f in self.frame:
            f.close()

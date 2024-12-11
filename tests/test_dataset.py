"""Tests for the dataset module.

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

import pytest
from pathlib import Path
import torch
from chuchichaestli.data.dataset import HDF5Dataset


# At the beginning of each pytest session 3 files test_{1,2,3}D.hdf5 are generated (see conftest.py)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_HDF5Dataset_init(dimensions):
    """Test the HDF5Dataset module."""
    f = Path(f"test_{dimensions}D.hdf5")
    ds = HDF5Dataset(f)
    assert len(ds) > 0
    assert hasattr(ds, "index")
    ds.purge_cache()


@pytest.mark.parametrize(
    "dimensions,index,meta_groups,scheme",
    [
        (1, 0, "**/metadata/*", "bijective"),
        (2, 100, "**/metadata/*", "bijective"),
        (3, 150, "**/metadata/*", "bijective"),
        (1, 0, "**/metadata/*", "collective"),
        (2, 100, "**/metadata/*", "collective"),
        (3, 150, "**/metadata/*", "collective"),
    ],
)
def test_HDF5Dataset_NotImplementedYet(dimensions, index, meta_groups, scheme):
    """Test the HDF5Dataset module."""
    f = Path(f"test_{dimensions}D.hdf5")
    with pytest.raises(NotImplementedError):
        ds = HDF5Dataset(f, meta_groups=meta_groups, scheme=scheme)
        ds.purge_cache()


@pytest.mark.parametrize(
    "dimensions,index,meta_groups,scheme",
    [
        (1, 0, None, None),
        (2, 100, None, None),
        (3, 150, None, None),
        (1, 0, "**/metadata/*", "analog"),
        (2, 100, "**/metadata/*", "analog"),
        (3, 150, "**/metadata/*", "analog"),
        (1, 0, "**/metadata/*", "surjective"),
        (2, 100, "**/metadata/*", "surjective"),
        (3, 150, "**/metadata/*", "surjective"),
    ],
)
def test_HDF5Dataset_getitem(dimensions, index, meta_groups, scheme):
    """Test the __getitem__ method from the HDF5Dataset module."""
    f = Path(f"test_{dimensions}D.hdf5")
    ds = HDF5Dataset(f, meta_groups=meta_groups, scheme=scheme)
    item = ds[index]
    assert isinstance(item[0], torch.Tensor)
    assert len(item[0].shape) == dimensions
    assert isinstance(item[1], dict)
    if scheme:
        assert ds.scheme == scheme
    elif meta_groups is None:
        assert ds.scheme == "analog"
    if meta_groups is None:
        assert item[1] == {}
    ds.purge_cache()


@pytest.mark.parametrize(
    "index,meta_groups,scheme",
    [
        (0, "**/metadata/*", "surjective"),
        (100, "**/metadata/*", "surjective"),
        (150, "**/metadata/*", "surjective"),
        (150, None, None)
    ],
)
def test_HDF5Dataset_getitem_no_collate(index, meta_groups, scheme):
    """Test the HDF5Dataset module."""
    ds = HDF5Dataset(
        Path("."),
        file_key="test_*.hdf5",
        meta_groups=meta_groups,
        scheme=scheme,
        collate=False,
        cache=False,
    )
    item = ds[index]
    assert len(item[0]) == 3
    assert isinstance(item[0][0], torch.Tensor)
    assert len(item[1]) == 3
    assert isinstance(item[1][0], dict)
    if scheme:
        assert ds.scheme == scheme
    elif meta_groups is None:
        assert ds.scheme == "analog"
    if meta_groups is None:
        assert item[1] == ({}, {}, {})
    ds.purge_cache()


@pytest.mark.parametrize(
    "index,meta_groups,scheme",
    [
        (1, "**/metadata/*", "surjective"),
        (100, "**/metadata/*", "surjective"),
        (150, "**/metadata/*", "surjective"),
        (150, None, None)
    ],
)
def test_HDF5Dataset_caching(index, meta_groups, scheme):
    """Test the HDF5Dataset module."""
    ds = HDF5Dataset(
        Path("."),
        file_key="test_*.hdf5",
        meta_groups=meta_groups,
        scheme=scheme,
        cache=4,
    )
    cache_sets = sum(s == 1 for s in ds.cache[0].states)
    ds[index]
    ds[index + 1]
    ds[index + 2]
    ds[index]
    assert cache_sets + 3 == sum(s == 1 for s in ds.cache[0].states)
    ds.purge_cache()


@pytest.mark.parametrize(
    "index,meta_groups,scheme",
    [
        (1, "**/metadata/*", "surjective"),
        (100, "**/metadata/*", "surjective"),
        (150, "**/metadata/*", "surjective"),
        (197, "**/metadata/*", "surjective"),
        (150, None, None)
    ],
)
def test_HDF5Dataset_no_collate_caching(index, meta_groups, scheme):
    """Test the HDF5Dataset module."""
    ds = HDF5Dataset(
        Path("."),
        file_key="test_*.hdf5",
        meta_groups=meta_groups,
        scheme=scheme,
        cache=4,
        collate=False,
    )
    cache_sets = sum(s == 1 for s in ds.cache[0].states)
    ds[index]
    ds[index + 1]
    ds[index + 2]
    ds[index]
    assert cache_sets + 9 == sum(s == 1 for s in ds.cache[0].states)
    ds.purge_cache()


@pytest.mark.parametrize(
    "preload,meta_groups,scheme",
    [
        (False, "**/metadata/*", "surjective"),
        (True, "**/metadata/*", "surjective"),
    ],
)
def test_HDF5Dataset_preload_timing(preload, meta_groups, scheme):
    """Test the HDF5Dataset module."""
    import time

    ds = HDF5Dataset(
        Path("."),
        file_key="test_*.hdf5",
        meta_groups=meta_groups,
        scheme=scheme,
        cache=4,
        collate=True,
        preload=preload,
    )
    t_ini = time.time()
    for i in range(len(ds)):
        ds[i]
    t_fin = time.time()
    print(f"Dataset iteration timing {preload=}: {t_fin - t_ini} s")
    ds.purge_cache()

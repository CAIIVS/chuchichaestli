"""Test configurations.

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

import os
from pathlib import Path
import h5py
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


def generate_sequential_hdf5(
    filename: str | Path | None = None,
    samples: int = 100,
    dimensions: int = 2,
    spatial_dim: int = 64,
    num_datasets: int = 1,
):
    """Generate an HDF5 dataset with sequential data."""
    if filename is None:
        filename = f"test_{dimensions}D.hdf5"
    shape = (spatial_dim,) * dimensions
    num_elem = samples * np.prod(shape)
    flat_seq = np.arange(num_elem, dtype=np.uint32)
    arr = flat_seq.reshape(samples, *shape)
    with h5py.File(filename, "w") as f:
        g = f.create_group("some/path_0")
        g.create_dataset("images", data=arr)
        g.attrs["generic_attr"] = "global_metadata"
        for i in range(samples):
            m = g.create_group(f"metadata/{i}")
            m.attrs["dim"] = f"{dimensions}"
            m.attrs["index"] = f"{i}"
            m.attrs["pointer_elem"] = f"{samples*i}"
            m.attrs["src_method"] = "np.arange"
        for i in range(num_datasets-1):
            g2 = f.create_group(f"some/other/path_{i+1}")
            data = arr[:samples//2, :, :] + (i+1)*num_elem
            g2.create_dataset("images", data=data)
    return f


# def generate_grouped_attrs(
#     filename: str | Path | None = None,
#     samples: int = 100,
#     dimensions: int = 2,
#     spatial_dim: int = 64,
#     num_datasets: int = 1,
#     num_attrsets: int = 1,
#     num_attrs: int = 100,
# ):
#     """Generate an HDF5 dataset with corresponding HDF5 attrs."""
#     if filename is None:
#         filename = f"test_{dimensions}D_attrs.hdf5"
#     shape = (spatial_dim,) * dimensions
#     num_elem = samples * np.prod(shape)
#     flat_seq = np.arange(num_elem, dtype=np.uint32)
#     arr = flat_seq.reshape(samples, *shape)
#     with h5py.File(filename, "w") as f:
#         g = f.create_group("some/path_0")
#         g.create_dataset("images", data=arr)
#         for i in range(samples):
#             m = g.create_group(f"metadata/{i}")
#             m.attrs["dim"] = f"{dimensions}"
#             m.attrs["pointer_elem"] = f"{samples*i}"
#             m.attrs["src_method"] = "np.arange"
#         for i in range(num_datasets-1):
#             g2 = f.create_group(f"some/other/path_{i+1}")
#             data = arr[:samples//2, :, :] + (i+1)*num_elem
#             g2.create_dataset("images", data=data)
#     return f


def generate_random_hdf5(
    filename: str | Path | None = None,
    samples: int = 100,
    dimensions: int = 2,
    spatial_dim: int = 64,
    num_datasets: int = 1,
):
    """Generate an HDF5 dataset with random data."""
    if filename is None:
        filename = f"rand_test_{dimensions}D.hdf5"
    shape = (spatial_dim,) * dimensions
    arr = np.random.randn(samples, *shape)
    with h5py.File(filename, "w") as f:
        g = f.create_group("some/path")
        g.create_dataset("images", data=arr)
        g.attrs["generic_attr"] = "global_metadata"
        for i in range(samples):
            m = g.create_group(f"metadata/{i}")
            m.attrs["foo"] = f"bar_{i}"
            m.attrs["src_method"] = "np.random.randn"
        for i in range(num_datasets-1):
            g2 = f.create_group(f"some/other/path_{i+1}")
            data = arr[:samples//2, :, :] + (i+1)*arr.size
            g2.create_dataset("images", data=data)
    return f


def pytest_sessionstart(session):
    """Actions before the tests."""
    for dimensions in [1, 2, 3]:
        if dimensions == 2:
            generate_sequential_hdf5(dimensions=dimensions, num_datasets=2)
            # generate_random_hdf5(dimensions=dimensions, num_datasets=2)
        else:
            generate_sequential_hdf5(dimensions=dimensions)
            # generate_random_hdf5(dimensions=dimensions)

def pytest_sessionfinish(session):
    """Actions after the tests."""
    for dimensions in [1, 2, 3]:
        f = Path(f"test_{dimensions}D.hdf5")
        os.remove(f)
        # f = Path(f"rand_test_{dimensions}D.hdf5")
        # os.remove(f)

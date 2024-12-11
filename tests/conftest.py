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


def generate_random_hdf5(
    filename: str | Path | None = None,
    samples: int = 200,
    dimensions: int = 2,
    spatial_dim: int = 64,
):
    """Generate an HDF5 dataset with random data."""
    if filename is None:
        filename = f"test_{dimensions}D.hdf5"
    arr = np.random.randn(samples, *((spatial_dim,) * dimensions))
    with h5py.File(filename, "w") as f:
        g = f.create_group("some/path")
        g.create_dataset("images", data=arr)
        for i in range(samples):
            m = g.create_group(f"metadata/{i}")
            m.attrs["foo"] = f"bar_{i}"
            m.attrs["src"] = "np.random.randn"
    return f


def pytest_sessionstart(session):
    """Actions before the tests."""
    for dimensions in [1, 2, 3]:
        generate_random_hdf5(dimensions=dimensions)

def pytest_sessionfinish(session):
    """Actions after the tests."""
    for dimensions in [1, 2, 3]:
        f = Path(f"test_{dimensions}D.hdf5")
        os.remove(f)

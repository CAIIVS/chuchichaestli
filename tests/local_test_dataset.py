"""Tests for local datasets.

This file does not contain actual unit test, but rather example usagee on
local datasets (saved locally in ./data); run with
`pytest tests/local_test_dataset.py[::<test_name>] -sv`
to inspect outputs and check correct file parsing by eye.

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
import gc
import time
import tqdm
import torch
from pathlib import Path
from chuchichaestli.data.dataset import HDF5Dataset, H5PyAttrs
from chuchichaestli.data.cache import nbytes


@pytest.mark.parametrize(
    "paths,groups,attr_groups,collate,parallel,pair,squash,attrs_retrieval",
    [
        [
            "./data/240818_tng50-1_dm*.hdf5",
            "/dm/*",
            "/dm/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/240818_tng50-1_dm_50_gids.0000.1000.hdf5",
            "*",
            None,
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            [
                "./data/240818_tng50-1_dm_50_gids.*.hdf5",
                "./data/240818_tng50-1_star_50_gids.*.hdf5",
            ],
            "*",
            None,
            False,
            False,
            True,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_1/mnist_test_1.h5",
            "/image",
            "/label",
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_2/mnist_test_2.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_3/mnist_test_3.h5",
            "**/image",
            "**/label",
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_4/mnist_test_4.h5",
            "/image/*",
            "/label/*",
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_5/train/mnist_test_*.h5",
            "/image",
            "/label",
            True,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_5/test/mnist_test_*.h5",
            "/image",
            "/label",
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_6/train/mnist_test_*.h5",
            "**/image",
            "**/label",
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_7/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_8/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_9/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_10/mnist_test_10.h5",
            "/image/*",
            ["/label/*", "/mask/*"],
            False,
            False,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_11/train/mnist_test_*.h5",
            "/image/*",
            ["/label/*", "/mask/*"],
            False,
            True,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_12/train/mnist_test_*.h5",
            "/image/*",
            "/blurred_image/*",
            False,
            True,
            False,
            True,
            "auto",
        ],
        [
            "./data/mnist_h5_tests/test_scenario_13/train/mnist_test_*.h5",
            "/image/*",
            "*",
            False,
            True,
            False,
            True,
            "auto",
        ],
    ],
)
def test_HDF5Dataset_init(
    paths, groups, attr_groups, collate, parallel, pair, squash, attrs_retrieval
):
    """Test the HDF5Dataset.__init__ method on local datasets in ./data."""
    h5 = HDF5Dataset(
        paths,
        groups,
        attr_groups=attr_groups,
        collate=collate,
        parallel=parallel,
        pair=pair,
        squash=squash,
    )
    if h5.frame:
        print("\n")
        h5.info(show_attrs_info=(h5.n_attrs < 20))
    h5.close()


@pytest.mark.parametrize(
    "paths,groups,attr_groups,collate,parallel,pair,squash,attrs_retrieval,index",
    [
        [
            "./data/240818_tng50-1_dm*.hdf5",
            "/dm/*",
            "/dm/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
        ],
        [
            [
                "./data/240818_tng50-1_dm_50_gids.*.hdf5",
                "./data/240818_tng50-1_star_50_gids.*.hdf5",
            ],
            "*",
            None,
            False,
            False,
            True,
            True,
            "auto",
            10,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_1/mnist_test_1.h5",
            "/image",
            "/label",
            False,
            False,
            False,
            True,
            "auto",
            1,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_2/mnist_test_2.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            2,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_3/mnist_test_3.h5",
            "**/image",
            "**/label",
            False,
            False,
            False,
            True,
            "auto",
            3,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_4/mnist_test_4.h5",
            "/image/*",
            "/label/*",
            False,
            False,
            False,
            True,
            "auto",
            4,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_5/train/mnist_test_*.h5",
            "/image",
            "/label",
            True,
            False,
            False,
            True,
            "auto",
            5,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_6/train/mnist_test_*.h5",
            "**/image",
            "**/label",
            False,
            False,
            False,
            True,
            "auto",
            6,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_7/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            7,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_8/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            1,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_9/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            5,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_10/mnist_test_10.h5",
            "/image/*",
            ["/label/*", "/mask/*"],
            False,
            False,
            False,
            True,
            "auto",
            16 * 1,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_11/train/mnist_test_*.h5",
            "/image/*",
            ["/label/*", "/mask/*"],
            False,
            True,
            False,
            True,
            "auto",
            8,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_12/train/mnist_test_*.h5",
            "/image/*",
            "/blurred_image/*",
            False,
            True,
            False,
            True,
            "auto",
            1,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_13/train/mnist_test_*.h5",
            "/image/*",
            "*",
            False,
            True,
            False,
            True,
            "auto",
            5,
        ],
        ["test_3D.hdf5", "*", "*", False, False, False, True, "auto", 28],
        [
            "test_3D.hdf5",
            "*images",
            "/some/path_0",
            False,
            False,
            False,
            True,
            "auto",
            1,
        ],
        [
            "test_2D.hdf5",
            "*images",
            "/some/path_0",
            False,
            False,
            False,
            True,
            "auto",
            1,
        ],
    ],
)
def test_HDF5Dataset_single_index(
    paths, groups, attr_groups, collate, parallel, pair, squash, attrs_retrieval, index
):
    """Test the __getitem__ method from the HDF5Dataset module (including attrs) on local datasets in ./data."""
    h5 = HDF5Dataset(
        paths,
        groups,
        attr_groups=attr_groups,
        collate=collate,
        parallel=parallel,
        pair=pair,
        squash=squash,
    )

    if h5.frame:
        print("\n")
        # h5.info(show_attrs_info=(h5.n_attrs <= 100))
        h5.info(show_attrs_info=False)
    out = h5[index]
    if not parallel and h5.n_attrs and attr_groups:
        if isinstance(out, torch.Tensor):
            print("Data:", out.shape)
        else:
            print("Data:", out[0].shape)
        if len(out) > 1:
            if isinstance(out[1], torch.Tensor) and len(out[1].shape) > 1:
                print("Attr:", out[1].shape)
            elif isinstance(out[1], torch.Tensor | H5PyAttrs):
                print("Attr:", out[1] if isinstance(out[1], torch.Tensor) else out[1])
            else:
                print(
                    "Attr:",
                    [a.shape if isinstance(a, torch.Tensor) else a for a in out[1]],
                )
    elif parallel and h5.n_attrs and attr_groups:
        data = out[0]
        attr = out[1]
        print("Data: ", [d.shape for d in data])
        print("Attr: ", [a.shape for a in attr])
    else:
        if isinstance(out, torch.Tensor):
            print("Data:", out.shape)
        else:
            print(out)
    h5.close()


@pytest.mark.parametrize(
    "paths,groups,attr_groups,collate,parallel,pair,squash,attrs_retrieval,ini,end",
    [
        [
            "./data/mnist_h5_tests/test_scenario_1/mnist_test_1.h5",
            "/image",
            "/label",
            False,
            False,
            False,
            True,
            "auto",
            0,
            10,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_2/mnist_test_2.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            0,
            10,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_3/mnist_test_3.h5",
            "**/image",
            "**/label",
            False,
            False,
            False,
            True,
            "auto",
            0,
            10,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_4/mnist_test_4.h5",
            "/image/*",
            "/label/*",
            False,
            False,
            False,
            True,
            "auto",
            0,
            10,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_5/train/mnist_test_*.h5",
            "/image",
            "/label",
            True,
            False,
            False,
            True,
            "auto",
            0,
            15,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_6/train/mnist_test_*.h5",
            "**/image",
            "**/label",
            False,
            False,
            False,
            True,
            "auto",
            0,
            15,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_7/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            0,
            15,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_8/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            0,
            5,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_9/train/mnist_test_*.h5",
            "*",
            "*",  # could also be None as no attribute sets are present
            False,
            False,
            False,
            True,
            "auto",
            0,
            60,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_10/mnist_test_10.h5",
            "/image/*",
            ["/label/*", "/mask/*"],
            False,
            False,
            False,
            True,
            "auto",
            0,
            160,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_11/train/mnist_test_*.h5",
            "/image/*",
            ["/label/*"],
            False,
            True,
            False,
            True,
            "auto",
            0,
            48,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_12/train/mnist_test_*.h5",
            "/image/*",
            None,
            False,
            True,
            False,
            True,
            "auto",
            0,
            3,
        ],
        [
            "./data/mnist_h5_tests/test_scenario_13/train/mnist_test_*.h5",
            "/image/*",
            "*",
            False,
            True,
            False,
            True,
            "auto",
            0,
            48,
        ],
    ],
)
def test_HDF5Dataset_loop_mnist_h5(
    paths,
    groups,
    attr_groups,
    collate,
    parallel,
    pair,
    squash,
    attrs_retrieval,
    ini,
    end,
):
    """Test the __getitem__ method from the HDF5Dataset module for a range of indices on local datasets in ./data."""
    h5 = HDF5Dataset(
        paths,
        groups,
        attr_groups=attr_groups,
        collate=collate,
        parallel=parallel,
        pair=pair,
        squash=squash,
    )

    if h5.frame:
        print("\n")
        h5.info(show_attrs_info=False)
    for i in range(ini, end):
        out = h5[i]
        if parallel:
            if h5.n_attrs != 0:  # mnist_h5_tests/test_scenario_11
                data = out[0]
                attr = out[1]
                if len(attr[0].size()) > 0:  # mnist_h5_tests/test_scenario_13
                    data = out[0]
                    print("Labels:", [t.flatten()[0].item() for t in data])
                    for t in data:
                        assert t.flatten()[0].item() == (i // 16)
                    continue
                print(
                    "Labels:", [a.item() for t, a in zip(data, attr) if a is not None]
                )
                for t, a in zip(data, attr):
                    if a is not None:
                        assert t.flatten()[0].item() == a.item()
            else:  # mnist_h5_tests/test_scenario_12
                data = out
                print("Labels:", [t.flatten()[0].item() for t in data])
                for t in data:
                    assert t.flatten()[0].item() == i
        elif isinstance(out, tuple | list) and isinstance(
            out[1], tuple | list
        ):  # mnist_h5_tests/test_scenario_10
            print("Label:", out[1][0].item())
            assert out[0].flatten()[0].item() == out[1][0].item()
        elif isinstance(out, tuple | list):  # mnist_h5_tests/test_scenario_[1,3,4,5,6]
            print("Label:", out[1].item())
            assert out[0].flatten()[0].item() == out[1].item()
        elif isinstance(out, torch.Tensor):  # mnist_h5_tests/test_scenario_[2,7,8,9]
            print("Pointer element:", out.flatten()[0].item())
            assert (i % 10) == out.flatten()[0].item()
    h5.close()


@pytest.mark.parametrize(
    "paths,groups,attr_groups,collate,parallel,pair,squash,attrs_retrieval,ini,end",
    [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            0,
            8,
        ],
        [
            [
                "./data/240818_tng50-1_dm_50_gids.*.hdf5",
                "./data/240818_tng50-1_star_50_gids.*.hdf5",
            ],
            "*",
            "*/metadata/*",
            False,
            False,
            True,
            True,
            "auto",
            0,
            8,
        ],
    ],
)
def test_HDF5Dataset_loop_skais_data(
    paths,
    groups,
    attr_groups,
    collate,
    parallel,
    pair,
    squash,
    attrs_retrieval,
    ini,
    end,
):
    """Test the __getitem__ method from the HDF5Dataset module for a range of indices on local datasets in ./data."""
    h5 = HDF5Dataset(
        paths,
        groups,
        attr_groups=attr_groups,
        collate=collate,
        parallel=parallel,
        pair=pair,
        squash=squash,
        cache=True,
        attrs_cache=True,
    )

    if h5.frame:
        print("\n")
        h5.info(show_attrs_info=False)
    for i in range(ini, end):
        out = h5[i]
        if len(h5.indexed_dims) > 1:
            print(out[0][0].shape, out[0][1].shape)
            print(out[1][0]["name"], out[1][1]["name"])
        else:
            print(out[0].shape)
            print(out[1]["name"])
    h5.close()


@pytest.mark.parametrize(
    "paths,groups,attr_groups,collate,parallel,pair,squash,attrs_retrieval,index",
    [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            None,
            False,
            False,
            False,
            True,
            "auto",
            0,
        ],
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            0,
        ],
        [
            [
                "./data/240818_tng50-1_dm_50_gids.*.hdf5",
                "./data/240818_tng50-1_star_50_gids.*.hdf5",
            ],
            "*",
            "*/metadata/*",
            False,
            False,
            True,
            True,
            "auto",
            0,
        ],
    ],
)
def test_HDF5Dataset_cache_skais_data(
    paths, groups, attr_groups, collate, parallel, pair, squash, attrs_retrieval, index
):
    """Test the __getitem__ method from the HDF5Dataset module for a range of indices on local datasets in ./data."""
    h5 = HDF5Dataset(
        paths,
        groups,
        attr_groups=attr_groups,
        collate=collate,
        parallel=parallel,
        pair=pair,
        squash=squash,
        cache=True,
        attrs_cache=True,
    )

    out = h5[index]
    out2 = h5[index]
    out3 = h5[index]
    if isinstance(out[0], torch.Tensor):
        assert torch.equal(out[0], out2[0])
        assert torch.equal(out2[0], out3[0])
        assert h5.cached_items == 1
    else:
        for t1, t2, t3 in zip(out[0], out2[0], out3[0]):
            assert torch.equal(t1, t2)
            assert torch.equal(t2, t3)
        assert h5.cached_items == 2

    print("\n")
    h5.info(show_attrs_info=False)
    print(h5.cache)
    print("Cached items:", h5.cached_items)
    print("Cached attrs:", h5.cached_attrs)
    h5.close()


@pytest.mark.parametrize(
    "paths,groups,attr_groups,collate,parallel,pair,squash,attrs_retrieval,subset,cache_size",
    [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            None,
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "8G",
        ]
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            None,
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "4G",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            None,
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "3G",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "2G",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "1G",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "800M",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "600M",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "400M",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "200M",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "100M",
        ],
    ]
    * 3
    + [
        [
            "./data/240818_tng50-1_dm_50_gids.*.hdf5",
            "*",
            "*/metadata/*",
            False,
            False,
            False,
            True,
            "auto",
            8000,
            "50M",
        ],
    ]
    * 3,
)
def measure_HDF5Dataset_skais_speedup(
    paths,
    groups,
    attr_groups,
    collate,
    parallel,
    pair,
    squash,
    attrs_retrieval,
    subset,
    cache_size,
):
    """Test an entire epoch of the skais dataset."""
    log_file = Path("hdf5dataset_cache_measurements.txt")
    if not log_file.exists():
        header = "# Time (uncached, in sec), Time (cached, in sec), Cache size (in bytes), Used cache (in bytes), Number of samples, Sample size (in bytes), Dataset size (in bytes)\n"
        with log_file.open(mode="w") as f:
            f.write(header)
    h5 = HDF5Dataset(
        paths,
        groups,
        attr_groups=attr_groups,
        collate=collate,
        parallel=parallel,
        pair=pair,
        squash=squash,
        cache=cache_size,
    )
    # indices = random.sample(range(len(h5)), subset)
    indices = range(subset)
    t_orig_ini = time.time()
    for i in tqdm.tqdm(indices):
        item = h5[i]
        len(item)
    t_orig_fin = time.time()
    t_orig = t_orig_fin - t_orig_ini
    if h5.frame:
        print("\n")
        h5.info(show_attrs_info=False)
    t_cache_ini = time.time()
    for i in tqdm.tqdm(indices):
        item = h5[i]
        len(item)
    t_cache_fin = time.time()
    t_cache = t_cache_fin - t_cache_ini
    print("Original time:", t_orig)
    print(f"Cached time ({cache_size}):", t_cache)
    print("Speed up:", (t_orig_fin - t_orig_ini) / (t_cache_fin - t_cache_ini))
    with log_file.open(mode="a+") as f:
        f.write(
            f"{t_orig:4.6f}, {t_cache:4.6f}, {int(h5.cache_size)}, {int(h5.cached_bytes)}, {subset}, {int(h5.sample_serial_size[0])}, {int(h5.serial_size[0])}\n"
        )
    h5.close()


def plot_speedup_measurements():
    """Plot the measurements in the speed-up log file."""
    import numpy as np
    from matplotlib import pyplot as plt

    log_file = Path("hdf5dataset_cache_measurements.txt")
    times = np.loadtxt(log_file, delimiter=",", usecols=(0, 1))
    cache_gb = np.loadtxt(
        log_file, delimiter=",", usecols=(2, 3, 5, 6), dtype=np.int64
    ) / (1 << 30)
    cache_b = np.loadtxt(log_file, delimiter=",", usecols=(5,), dtype=np.int64)
    other = np.loadtxt(log_file, delimiter=",", usecols=(4,), dtype=np.int64)
    plt.plot(
        cache_gb[:, 0], times[:, 0], marker="o", c="#DC7A91", lw=0, label="uncached"
    )
    plt.hlines(
        np.average(times[:, 0]),
        0,
        np.max(cache_gb[:, 0]),
        colors="#DC7A91",
        linestyles="dotted",
    )
    plt.plot(
        cache_gb[:, 0], times[:, 1], marker="o", c="#393E75", lw=0, label="(sto)cached"
    )
    # plt.xscale("log", base=2)
    plt.plot([], [], " ", label=f"Number of samples: {other[0]}")
    plt.plot([], [], " ", label=f"Sample size: {nbytes(cache_b[0]).as_str()}")
    plt.legend()
    plt.xlabel("Cache reserved [GiB]")
    plt.ylabel("Wall time [sec]")
    plt.tight_layout()
    plt.savefig("hdf5dataset_cache_measurements_timings.png", dpi=300)
    plt.close()

    speed_up = times[:, 0] / times[:, 1]
    plt.plot(
        cache_gb[:, 0], speed_up, marker="o", c="#393E75", lw=0, label="(sto)cached"
    )
    plt.hlines(
        1,
        0,
        np.max(cache_gb[:, 0]),
        colors="#DC7A91",
        linestyles="dotted",
        label="no speed-up",
    )
    plt.yscale("log", base=2)
    plt.xscale("log", base=2)
    plt.plot([], [], " ", label=f"Number of samples: {other[0]}")
    plt.plot([], [], " ", label=f"Sample size: {nbytes(cache_b[0]).as_str()}")
    plt.legend()
    plt.xlabel("Cache reserved [GiB]")
    plt.ylabel("Speed-up [cached/uncached]")
    plt.tight_layout()
    plt.savefig("hdf5dataset_cache_measurements_speedup.png", dpi=300)
    plt.close()

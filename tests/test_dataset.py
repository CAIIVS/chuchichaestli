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
from chuchichaestli.data.cache import nbytes

# At the beginning of each pytest session 3 files test_{1,2,3}D.hdf5 are generated (see conftest.py)


@pytest.mark.parametrize(
    "path",
    [
        "/data/dir/images*",
        "**/images*",
        ["/data/dir/images*"],
        ["/data/dir/images*", "/data/dir2/images*"],
        ["/data/dir/images*", "/data/dir2/images"],
    ],
)
def test_split_glob(path):
    """Test the HDF5Dataset._split_glob method."""
    p_out, k_out = HDF5Dataset._split_glob(path)
    assert len(p_out) == len(k_out)
    assert isinstance(p_out, list)
    assert isinstance(k_out, list)
    assert k_out is not None
    print("\n", p_out, k_out)


@pytest.mark.parametrize(
    "path",
    [
        "/data/dir/images",
        "/data/dir/images.hdf5",
        Path("/data/dm/images"),
    ],
)
def test_split_glob_invariant(path):
    """Test the HDF5Dataset._split_glob method."""
    p_out, k_out = HDF5Dataset._split_glob(path)
    assert isinstance(p_out, list)
    assert isinstance(k_out, list)
    assert len(p_out) == len(k_out)
    assert None in k_out
    print("\n", p_out, k_out)


@pytest.mark.parametrize(
    "paths",
    [
        "test_1D.hdf5",
        "test_2D.hdf5",
        Path("test_3D.hdf5"),
        "test_*D.hdf5",
        ["test_1D.hdf5", "test_3D.hdf5"],
        ["test_*D.hdf5", "test_3D.hdf5"],
        ("test_*D.hdf5",),
    ],
)
def test_glob_path(paths):
    """Test the HDF5Dataset.glob_path method."""
    files = HDF5Dataset.glob_path(paths)
    assert isinstance(files, list)
    assert all(isinstance(p, Path) for p in files)
    assert all([p.suffix in (".hdf5", ".h5") for p in files])
    print("\n", files)


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "paths,groups,collate,parallel",
    [
        ["test_1D.hdf5", "*", False, False],
        [["test_2D.hdf5", "test_2D.hdf5"], "*", True, True],
        [Path("test_3D.hdf5"), "*", False, False],
    ],
)
def test_HDF5Dataset_init(paths, groups, collate, parallel):
    """Test the HDF5Dataset.__init__ method."""
    h5 = HDF5Dataset(paths, groups, collate=collate, parallel=parallel)
    print(f"\n{h5}")
    print(h5.raw_dims)


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "paths,groups,collate,parallel",
    [
        ["test_1D.hdf5", "*", False, False],
        [["test_2D.hdf5", "test_2D.hdf5"], "*", True, True],
        [Path("test_3D.hdf5"), "*", False, False],
    ],
)
def test_HDF5Dataset_info(paths, groups, collate, parallel):
    """Test the HDF5Dataset.info method."""
    h5 = HDF5Dataset(paths, groups, collate=collate, parallel=parallel)
    print()
    info = h5.info()
    assert isinstance(info, str)


@pytest.mark.parametrize(
    "data",
    [
        [["d1", "d2"]],
        [["d1", "d2"], ["d3", "d4"], ["d5", "d6"]],
        [["d1"], ["d2"], ["d3"], ["d4"]],
        [["d1"], ["d2"], ["d3"], ["d4", "d5"]],
        [[(4000, 512, 512)], [(4000, 512, 512)], [(4000, 512, 512)]],
        [[(2000, 512, 512), (1000, 256, 256)], [(2000, 512, 512), (1000, 256, 256)]],
        [[None, "d2"]],
    ],
)
def test_HDF5Dataset__collate(data):
    """Test the HDF5Dataset._collate method."""
    c = HDF5Dataset._collate(data)
    maxlen = max(len(d) for d in data)
    assert len(c) <= maxlen
    assert len([dij for di in data for dij in di]) >= len(
        [cij for ci in c for cij in ci]
    )
    print(c)


@pytest.mark.parametrize(
    "data",
    [
        [["d1", "d2"]],
        [["d1", "d2"], ["d3", "d4"], ["d5", "d6"]],
        [["d1"], ["d2"], ["d3"], ["d4"]],
        [["d1"], ["d2"], ["d3"], ["d4", "d5"]],
        [[(4000, 512, 512)], [(4000, 512, 512)], [(4000, 512, 512)]],
        [[(2000, 512, 512), (1000, 256, 256)], [(2000, 512, 512), (1000, 256, 256)]],
        [[None, "d2"]],
    ],
)
def test_HDF5Dataset__collate_with_pad(data):
    """Test the HDF5Dataset._collate method (with padding)."""
    c = HDF5Dataset._collate(data, pad=True)
    maxlen = max(len(d) for d in data)
    assert len(c) == maxlen
    assert len([dij for di in data for dij in di]) <= len(
        [cij for ci in c for cij in ci]
    )
    print(c)


@pytest.mark.parametrize(
    "data",
    [
        [["d1", "d2"]],
        [["d1", "d2"], ["d3", "d4"], ["d5", "d6"]],
        [["d1"], ["d2"], ["d3"], ["d4"]],
        [["d1"], ["d2"], ["d3"], ["d4", "d5"]],
        [[(4000, 512, 512)], [(4000, 512, 512)], [(4000, 512, 512)]],
        [[(2000, 512, 512), (1000, 256, 256)], [(2000, 512, 512), (1000, 256, 256)]],
        [[None, "d2"]],
    ],
)
def test_HDF5Dataset__contiguate(data):
    """Test the HDF5Dataset._contiguate method."""
    c = HDF5Dataset._contiguate(data)
    assert len(c) == 1
    print(c)


@pytest.mark.parametrize(
    "data",
    [
        [["d1", "d2"]],
        [["d1", "d2"], ["d3", "d4"], ["d5", "d6"]],
        [["d1"], ["d2"], ["d3"], ["d4"]],
        [["d1"], ["d2"], ["d3"], ["d4", "d5"]],
        [[(4000, 512, 512)], [(4000, 512, 512)], [(4000, 512, 512)]],
        [[(2000, 512, 512), (1000, 256, 256)], [(2000, 512, 512), (1000, 256, 256)]],
        [[None, "d2"]],
    ],
)
def test_HDF5Dataset__parallelize(data):
    """Test the HDF5Dataset._parallelize method."""
    c = HDF5Dataset._parallelize(data)
    assert len(data) <= len(c)
    print(c)


@pytest.mark.parametrize(
    "data",
    [
        [["d11"], ["d12"], ["d21"], ["d22"]],
        [["d111", "d112"], ["d121", "d122"], ["d211", "d212"], ["d221", "d222"]],
    ],
)
def test_HDF5Dataset__pair(data):
    """Test the HDF5Dataset._pair method."""
    c = HDF5Dataset._pair(data)
    assert len(c) == 2
    print(c)


@pytest.mark.parametrize(
    "paths,groups",
    [
        ["test_1D.hdf5", "*"],
        [["test_2D.hdf5", "test_2D.hdf5"], "*"],
        [Path("test_3D.hdf5"), "*"],
    ],
)
def test_HDF5Dataset_dims(paths, groups):
    """Test the HDF5Dataset.dims property (w/o collation and w/ parallelization)."""
    h5 = HDF5Dataset(paths, groups, collate=False, parallel=False)
    assert isinstance(h5.dims, list | tuple)
    print(h5.dims)
    h5.close()


@pytest.mark.parametrize(
    "paths,groups",
    [
        [["test_1D.hdf5", "test_1D.hdf5"], "*"],  # no collation necessary
        [["test_2D.hdf5", "test_2D.hdf5"], "*"],
    ],
)
def test_HDF5Dataset_dims_collate(paths, groups):
    """Test the HDF5Dataset.dims property (w/ collation but w/o parallelization)."""
    h5 = HDF5Dataset(paths, groups, collate=True, parallel=False)
    if len(h5.dims) == 0:
        return
    assert isinstance(h5.dims, tuple)  # automatically contiguated
    print(h5.dims)
    h5.close()


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "paths,groups",
    [
        ["test_1D.hdf5", "*"],
        [["test_2D.hdf5", "test_2D.hdf5"], "*"],
        [[Path("test_2D.hdf5"), Path("test_3D.hdf5")], "*"],
    ],
)
def test_HDF5Dataset_dims_parallel(paths, groups):
    """Test the HDF5Dataset.dims property (w/o collation but w/ parallelization)."""
    h5 = HDF5Dataset(paths, groups, collate=False, parallel=True)
    if len(h5.dims) == 0:
        return
    assert len(h5.dims) >= 1  # automatically contiguated
    are_eq = [
        l11 == l22 for l1, l2 in zip(h5.dims, h5.raw_dims) for l11, l22 in zip(l1, l2)
    ]
    is_eq = all(are_eq) or h5.dims == h5.raw_dims
    assert is_eq
    print(h5.dims)
    h5.close()


@pytest.mark.parametrize(
    "paths,groups",
    [
        [[Path("test_2D.hdf5"), Path("test_3D.hdf5")], "*"],
    ],
)
def test_HDF5Dataset_dims_collate_with_valueerror(paths, groups):
    """Test the error mode of HDF5Dataset.dims property (w/ collation but w/o parallelization)."""
    with pytest.raises(ValueError):
        HDF5Dataset(paths, groups, collate=False, parallel=False, squash=True)


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "paths,groups",
    [
        [[Path("test_2D.hdf5"), Path("test_3D.hdf5")], "*"],
    ],
)
def test_HDF5Dataset_dims_parallel_with_warning(paths, groups):
    """Test the warning mode of HDF5Dataset.dims property (w/o collation but w/ parallelization)."""
    h5 = HDF5Dataset(paths, groups, collate=False, parallel=True)
    with pytest.warns(UserWarning):
        h5.dims
    h5.close()


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "paths,groups,collate,parallel,squash,expected",
    [
        ["test_1D.hdf5", "*", False, False, True, 100],
        [["test_3D.hdf5", "test_3D.hdf5"], "*", False, True, True, 100],
        [["test_3D.hdf5", "test_3D.hdf5"], "*", False, False, True, 200],
        [
            ["test_1D.hdf5", "test_1D.hdf5", "test_1D.hdf5"],
            "*",
            False,
            False,
            True,
            300,
        ],
        [["test_2D.hdf5", "test_2D.hdf5"], "*", False, True, True, 150],
        [["test_2D.hdf5", "test_2D.hdf5"], "*", True, False, True, 300],
        [[Path("test_2D.hdf5"), Path("test_3D.hdf5")], "*", False, True, True, 100],
        [[Path("test_2D.hdf5"), Path("test_3D.hdf5")], "*", False, False, False, 250],
    ],
)
def test_HDF5Dataset___len__(paths, groups, collate, parallel, squash, expected):
    """Test the __len__ method of the HDF5Dataset class."""
    h5 = HDF5Dataset(paths, groups, collate=collate, squash=squash, parallel=parallel)
    length = len(h5)
    assert length == expected
    print(length)
    h5.close()


@pytest.mark.parametrize(
    "dim,N_files,groups,index",
    [
        [1, 1, "*", 0],  # (100, 64)[0]   -> (0, 1, ..., 63)
        [1, 1, "*", 1],  # (100, 64)[1]   -> (64, 65, ..., 127)
        [1, 1, "*", 50],  # (100, 64)[100] -> (6400, 6401, ..., 6463)
        [1, 2, "*", 99],  # (200, 64)[199] -> (12736, 12737, ..., 12799)
        [1, 2, "*", 100],  # (200, 64)[200] -> (0, 1, ..., 64)
        [1, 1, "*", -1],  # (100, 64)[-1]  -> (12736, ..., 12799)
        [1, 2, "*", -1],  # (200, 64)[-1]  -> (12736, ..., 12799)
        [2, 2, ["some/path*", "some/other/path*"], 0],
        [2, 2, ["some/path*", "some/other/path*"], 1],
        [2, 2, ["some/path*", "some/other/path*"], 64],
        [2, 2, ["some/path*", "some/other/path*"], 147],
        [2, 2, ["some/path*", "some/other/path*"], 148],
        [2, 2, ["some/path*", "some/other/path*"], 149],
        [2, 2, ["some/path*", "some/other/path*"], 150],
        [2, 2, ["some/path*", "some/other/path*"], 151],
        [2, 1, ["some/path*", "some/other/path*"], -1],
        [2, 2, ["some/path*", "some/other/path*"], -1],
        [3, 2, "*", 0],
        [3, 2, "*", 1],
        [3, 2, "*", 2],
        [3, 2, "*", 64],
        [3, 2, "*", 98],
        [3, 2, "*", 99],
        [3, 2, "*", 100],
        [3, 2, "*", 101],
        [3, 2, "*", -1],
    ],
)
def test_HDF5Dataset___getitem__(dim, N_files, groups, index):
    """Test the __getitem__ method from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    dtype = (
        torch.float32 if dim < 3 else torch.float64
    )  # parsing int -> float fails for 3D otherwise
    h5 = HDF5Dataset(
        paths, groups, collate=False, squash=True, parallel=False, dtype=dtype
    )
    output = h5[index]
    file_samples = 100 if dim in [1, 3] else 150
    tot_samples = file_samples * (64**dim)
    item_first = output.flatten()[0].item()
    item_last = output.flatten()[-1].item()
    item_first_expect = (index * 64**dim) % tot_samples
    item_last_expect = (index * 64**dim + (64**dim) - 1) % tot_samples
    print(
        "\nN_files, dim, index, item(first), item(last), expected(first), expected(last)",
        f"\n{N_files}",
        f"{dim}d",
        index,
        item_first,
        item_last,
        item_first_expect,
        item_last_expect,
    )
    assert isinstance(output, torch.Tensor)
    assert item_first_expect == item_first
    assert item_last_expect == item_last
    h5.close()


@pytest.mark.parametrize(
    "dim,N_files,groups,index",
    [
        [1, 1, "*", 0],
        [1, 1, "*", -1],
        [3, 1, "*", 0],
        [3, 1, "*", 1],
        [3, 1, "*", 2],
        [3, 1, "*", 64],
        [3, 1, "*", 99],
        [3, 1, "*", -1],
    ],
)
def test_HDF5Dataset_parallel_single_dataset___getitem__(dim, N_files, groups, index):
    """Test the __getitem__ method from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    dtype = (
        torch.float32 if dim < 3 else torch.float64
    )  # parsing int -> float fails for 3D otherwise
    h5 = HDF5Dataset(
        paths, groups, collate=False, squash=True, parallel=True, dtype=dtype
    )
    output = h5[index]
    file_samples = 100 if dim in [1, 3] else 150
    tot_samples = file_samples * (64**dim)
    item_first = output.flatten()[0].item()
    item_last = output.flatten()[-1].item()
    item_first_expect = (index * 64**dim) % tot_samples
    item_last_expect = (index * 64**dim + (64**dim) - 1) % tot_samples
    print(
        "\nN_files, dim, index, item(first), item(last), expected(first), expected(last)",
        f"\n{N_files}",
        f"{dim}d",
        index,
        item_first,
        item_last,
        item_first_expect,
        item_last_expect,
    )
    assert isinstance(output, torch.Tensor)
    assert item_first_expect == item_first
    assert item_last_expect == item_last
    h5.close()


@pytest.mark.parametrize(
    "dim,N_files,groups,index",
    [
        [1, 1, "*", 0],  # 1x(100, 64)[0]   -> (0, 1, ..., 64)
        [1, 2, "*", 0],  # 2x(200, 64)[0]   -> (0, 1, ..., 64), (0, 1, ..., 64)
        [1, 3, "*", 0],  # 2x(200, 64)[0]   -> (0, 1, ..., 64), (0, 1, ..., 64)
        [1, 2, "*", 99],  # 2x(200, 64)[199] -> (0, 1, ..., 64), (0, 1, ..., 64)
        [1, 2, "*", 50],  # 2x(200, 64)[100] -> (12736, ..., 12799)
        [1, 2, "*", -1],  # 2x(200, 64)[-1]  -> (12736, ..., 12799)
        [2, 2, ["some/path*", "some/other/path*"], 0],
        [2, 2, ["some/path*", "some/other/path*"], 149],
        [3, 2, "*", 0],
        [3, 2, "*", 1],
        [3, 2, "*", 50],
        [3, 2, "*", 64],
        [3, 2, "*", -1],
    ],
)
def test_HDF5Dataset_parallel___getitem__(dim, N_files, groups, index):
    """Test the __getitem__ method from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    paths = [f"test_{dim}D.hdf5"] * N_files
    dtype = torch.float32 if dim < 3 else torch.float64
    h5 = HDF5Dataset(
        paths, groups, collate=False, squash=True, parallel=True, dtype=dtype
    )
    output = h5[index]
    file_samples = 100 if dim in [1, 3] else 150
    tot_samples = file_samples * (64**dim)
    item_first_expect = (index * 64**dim) % tot_samples
    item_last_expect = (index * 64**dim + (64**dim) - 1) % tot_samples
    if len(h5.frame_datasets) > 1:
        item_first = [o.flatten()[0].item() for o in output]
        item_last = [o.flatten()[-1].item() for o in output]
    else:
        item_first = output.flatten()[0].item()
        item_last = output.flatten()[-1].item()
    print(
        "\nN_files, dim, index, item(first), item(last), expected(first), expected(last)",
        f"\n{N_files}",
        f"{dim}d",
        index,
        item_first,
        item_last,
        item_first_expect,
        item_last_expect,
    )
    if N_files == len(h5.frame_datasets) == 1:
        assert isinstance(output, torch.Tensor)
        assert item_first_expect == item_first
        assert item_last_expect == item_last
    else:
        assert isinstance(output, tuple)
        assert all([torch.all(o == output[0]) for o in output])
        assert all([o.flatten()[0].item() == item_first_expect for o in output])
        assert all([o.flatten()[-1].item() == item_last_expect for o in output])
    h5.close()


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "dim,N_files,groups,index",
    [
        [2, 1, ["some/path*", "some/other/path*"], 0],
        [2, 1, ["some/path*", "some/other/path*"], 49],
        [2, 1, ["some/path*", "some/other/path*"], -1],
        [2, 1, ["some/path*", "some/other/path*"], -10],
    ],
)
def test_HDF5Dataset_parallel_single_file_multiple_ds__getitem__(
    dim, N_files, groups, index
):
    """Test the __getitem__ method from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    paths = [f"test_{dim}D.hdf5"] * N_files
    dtype = torch.float32 if dim < 3 else torch.float64
    h5 = HDF5Dataset(
        paths, groups, collate=False, squash=True, parallel=True, dtype=dtype
    )
    datasets = h5.make_index()
    print("\nDataset shapes", [d.shape for d in datasets])
    output = h5[index]
    file_samples = 100 if dim in [1, 3] else 150
    if index < 0:
        index = index % 50
    tot_samples = file_samples * (64**dim)
    ds1_first_expect = (index * 64**dim) % tot_samples
    ds1_last_expect = (index * 64**dim + (64**dim) - 1) % tot_samples
    ds2_first_expect = ((100 + index) * 64**dim) % tot_samples
    ds2_last_expect = ((100 + index) * 64**dim + (64**dim) - 1) % tot_samples
    item_first = [o.flatten()[0].item() for o in output]
    item_last = [o.flatten()[-1].item() for o in output]
    print(
        "\nN_files, dim, index, item(first), item(last), expected(first), expected(last)",
        f"\n{N_files}",
        f"{dim}d",
        index,
        item_first,
        item_last,
        [ds1_first_expect, ds1_last_expect],
        [ds2_first_expect, ds2_last_expect],
    )
    assert isinstance(output, tuple)
    assert output[0].flatten()[0].item() == ds1_first_expect
    assert output[0].flatten()[-1].item() == ds1_last_expect
    assert output[1].flatten()[0].item() == ds2_first_expect
    assert output[1].flatten()[-1].item() == ds2_last_expect
    h5.close()


@pytest.mark.parametrize(
    "dim,N_files,groups,index",
    [
        [1, 2, "*", 99],
        [1, 2, "*", 100],
        [1, 2, "*", -1],
        [2, 2, ["some/path*", "some/other/path*"], 0],
        [2, 2, ["some/path*", "some/other/path*"], 1],
        [2, 2, ["some/path*", "some/other/path*"], 49],
        [2, 2, ["some/path*", "some/other/path*"], 99],
        [2, 2, ["some/path*", "some/other/path*"], 100],
        [2, 2, ["some/path*", "some/other/path*"], 149],
        [2, 2, ["some/path*", "some/other/path*"], 150],
        [2, 2, ["some/path*", "some/other/path*"], 151],
        [2, 2, ["some/path*", "some/other/path*"], 199],
        [2, 2, ["some/path*", "some/other/path*"], 200],
        [2, 2, ["some/path*", "some/other/path*"], 249],
        [2, 2, ["some/path*", "some/other/path*"], 250],
        [2, 2, ["some/path*", "some/other/path*"], 299],
        [2, 2, ["some/path*", "some/other/path*"], -1],
        [2, 3, ["some/path*", "some/other/path*"], 449],
        [2, 5, ["some/path*", "some/other/path*"], 749],
        [3, 2, "*", 0],
        [3, 2, "*", 1],
        [3, 2, "*", 2],
        [3, 2, "*", 64],
        [3, 2, "*", 98],
        [3, 2, "*", 99],
        [3, 2, "*", 100],
        [3, 2, "*", 101],
        [3, 2, "*", -1],
        [3, 3, "*", 299],
    ],
)
def test_HDF5Dataset_collate___getitem__(dim, N_files, groups, index):
    """Test the __getitem__ method from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    paths = [f"test_{dim}D.hdf5"] * N_files
    dtype = torch.float32 if dim < 3 else torch.float64
    h5 = HDF5Dataset(
        paths, groups, collate=True, squash=True, parallel=False, dtype=dtype
    )
    output = h5[index]
    # due to collation at dim == 2
    if dim == 2:
        file_sections = (0,) + (100,) * N_files + (150,) * N_files
    elif dim in [1, 3]:
        file_sections = (0,) + (100,) * N_files
    file_intervals = [
        sum(el) for el in zip(*[file_sections[i:] for i in range(N_files)])
    ]
    index_section = [index < i for i in file_intervals].index(True)
    index_mod = index - file_sections[index_section]
    tot_samples = file_sections[-1] * 64**dim
    item_first_expect = ((index_mod) * 64**dim) % tot_samples
    item_last_expect = ((index_mod) * 64**dim + (64**dim) - 1) % tot_samples
    item_first = output.flatten()[0].item()
    item_last = output.flatten()[-1].item()
    print(
        "\nN_files, dim, index, item(first), item(last), expected(first), expected(last)",
        f"\n{N_files}",
        f"{dim}d",
        index,
        index_section,
        index_mod,
        item_first,
        item_last,
        item_first_expect,
        item_last_expect,
    )
    assert isinstance(output, torch.Tensor)
    assert item_first_expect == item_first
    assert item_last_expect == item_last
    h5.close()


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "dim,N_files,groups,cache_size",
    [
        [1, 1, ["some/path*", "some/other/path*"], "1G"],
        [1, 2, ["some/path*", "some/other/path*"], "1G"],
        [2, 1, ["some/path*", "some/other/path*"], "1G"],
        [3, 1, ["some/path*", "some/other/path*"], "1G"],
    ],
)
def test_HDF5Dataset_cache_init(dim, N_files, groups, cache_size):
    """Test the cache from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    paths = [f"test_{dim}D.hdf5"] * N_files
    dtype = torch.float32 if dim < 3 else torch.float64
    h5 = HDF5Dataset(paths, groups, squash=True, dtype=dtype, cache=cache_size)
    assert len(h5.cache[0]) == 1
    assert h5.cache_size == nbytes(cache_size)
    print(f"\n{h5.cache}")
    h5.close()


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "dim,N_files,groups,parallel,cache_size,index",
    [
        [1, 1, ["some/path*", "some/other/path*"], False, "1G", 0],
        [1, 2, ["some/path*", "some/other/path*"], False, "1G", 100],
        [1, 2, ["some/path*", "some/other/path*"], True, "1G", 99],
        [2, 1, ["some/path*", "some/other/path*"], False, "1G", 0],
        [3, 1, ["some/path*", "some/other/path*"], False, "1G", 0],
    ],
)
def test_HDF5Dataset_cache(dim, N_files, groups, parallel, cache_size, index):
    """Test the cache from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    paths = [f"test_{dim}D.hdf5"] * N_files
    dtype = torch.float32 if dim < 3 else torch.float64
    h5 = HDF5Dataset(
        paths, groups, squash=True, parallel=parallel, dtype=dtype, cache=cache_size
    )
    if parallel:
        assert len(h5.cache[0]) == N_files
    else:
        assert len(h5.cache[0]) == 1
    assert h5.cache_size == nbytes(cache_size)
    assert h5.cache[0][0][index] is None
    n_cached_before = h5.cached_items
    out1 = h5[index][0] if parallel else h5[index]
    n_cached_after = h5.cached_items
    out2 = h5[index][0] if parallel else h5[index]
    n_cached_afterafter = h5.cached_items
    assert n_cached_before < n_cached_after
    assert n_cached_after == n_cached_afterafter
    assert torch.equal(h5.cache[0][0][index], out1)
    assert torch.equal(out1, out2)
    print(f"\n{h5.cache}")
    h5.close()


@pytest.mark.filterwarnings("ignore:Dimensions of datasets")
@pytest.mark.parametrize(
    "dim,N_files,groups,parallel,cache_size,index",
    [
        [1, 1, ["some/path*", "some/other/path*"], False, "2G", 0],
        [1, 2, ["some/path*", "some/other/path*"], False, "2G", 100],
        [1, 2, ["some/path*", "some/other/path*"], True, "2G", 99],
        [2, 1, ["some/path*", "some/other/path*"], False, "2G", 0],
        [3, 1, ["some/path*", "some/other/path*"], False, "2G", 0],
    ],
)
def test_HDF5Dataset_preload(dim, N_files, groups, parallel, cache_size, index):
    """Test preloading from the HDF5Dataset module."""
    # Test files have sequential data -> test first and last elements of extracted tensor
    import time

    if N_files > 1:
        paths = [f"test_{dim}D.hdf5"] * N_files
    else:
        paths = f"test_{dim}D.hdf5"
    preload = True
    paths = [f"test_{dim}D.hdf5"] * N_files
    dtype = torch.float32 if dim < 3 else torch.float64
    t_ini = time.time()
    h5 = HDF5Dataset(
        paths,
        groups,
        squash=True,
        parallel=parallel,
        dtype=dtype,
        cache=cache_size,
        preload=preload,
    )
    t_fin = time.time()
    print(f"\n{h5.cache}")
    print(f"Dataset iteration time {preload=}: {t_fin - t_ini:6.4f} s")
    h5.close()

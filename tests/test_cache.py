"""Tests for the cache module.

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
import numpy as np
import torch
from chuchichaestli.data.cache import (
    nbytes,
    get_max_ram,
    get_max_shm,
    serial_byte_size,
    SharedArray,
    SharedDict,
    SharedDictList,
)


@pytest.mark.parametrize("x", ["2.0G", "2.0GB", "2.0 GB", 2147483648.0, 2147483648])
def test_nbytes_2G(x):
    """Test the nbytes class."""
    b = nbytes(x)
    assert 0 < b < 10**10
    assert isinstance(b, nbytes)
    assert isinstance(b, float)
    assert b == float(b)
    assert isinstance(b.as_str(), str)
    assert isinstance(b.as_bstr(), str)
    assert isinstance(b.to("G"), nbytes)
    assert isinstance(b.to("G"), float)


@pytest.mark.parametrize("x", [None, 0.0, 0, "GB"])
def test_nbytes_null(x):
    """Test the nbytes class in edge cases."""
    b = nbytes(x)
    assert b == 0
    assert isinstance(b, nbytes)
    assert isinstance(b, float)
    assert b == float(b)
    assert isinstance(b.as_str(), str)
    assert isinstance(b.as_bstr(), str)
    assert isinstance(b.to("G"), nbytes)
    assert isinstance(b.to("G"), float)


def test_get_max_ram():
    """Test the get_max_ram function."""
    ram_size = get_max_ram()
    assert isinstance(ram_size, float)
    assert ram_size > 0
    print(ram_size)


def test_get_max_shm():
    """Test the get_max_shm function."""
    shm_size = get_max_shm()
    assert isinstance(shm_size, float)
    assert shm_size > 0
    print(shm_size)


def test_serial_byte_size():
    """Test the serial_byte_size function."""
    test_dict = {
        "numbers": [1, 2, 3, 4, 5],
        "index": 1,
        "bool": False,
        "ratio": 0.8,
        "foo": "bar",
    }
    dct_size = serial_byte_size(test_dict)
    assert dct_size == 87


def test_serial_byte_size_empty():
    """Test the serial_byte_size function."""
    dct_size = serial_byte_size({})
    assert dct_size == 5


@pytest.mark.parametrize(
    "shape,dtype,cache_size",
    [
        ((100, 1, 64, 64), torch.float32, 1.0),
        ((200, 1, 32, 32), torch.float64, "4G"),
        ((300, 1, 32, 32), torch.int8, 4),
    ],
)
def test_SharedArray_init(shape, dtype, cache_size):
    """Test the SharedArray module."""
    cache = SharedArray(
        shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
    )
    assert isinstance(cache.array, torch.Tensor)
    assert isinstance(cache.states, torch.Tensor)
    assert len(cache) == shape[0]
    assert cache.get_state(0)[0].value == 0


@pytest.mark.parametrize(
    "shape,dtype,cache_size",
    [
        ((300, 1, 32, 32), torch.int16, "40KB"),
        ((200, 1, 32, 32), torch.float32, "4B"),
    ],
)
def test_SharedArray_ooc(shape, dtype, cache_size):
    """Test the SharedArray module with out-of-cache values."""
    print()
    cache = SharedArray(
        shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
    )
    assert isinstance(cache.array, torch.Tensor)
    assert isinstance(cache.states, torch.Tensor)
    assert len(cache) < shape[0]
    assert cache.get_state(-1)[0].value == 2


@pytest.mark.parametrize(
    "shape,dtype,cache_size,setindex,getindex",
    [
        ((100, 1, 64, 64), torch.float32, 1.0, 10, 10),
        ((100, 1, 64, 64), torch.float32, 1.0, 10, 12),
    ],
)
def test_SharedArray_setitem_and_getitem(
    shape,
    dtype,
    cache_size,
    setindex,
    getindex,
):
    """Test the SharedArray module's setitem method."""
    print()
    cache = SharedArray(
        shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
    )
    cache[setindex] = torch.randn(*shape[1:])
    if getindex == setindex:
        assert cache[getindex] is not None
        assert isinstance(cache[getindex], torch.Tensor)
    else:
        assert cache[getindex] is None


def test_SharedArray_clear_index(
    shape=(100, 1, 64, 64),
    dtype=torch.float32,
    cache_size=1.0,
    index=10,
):
    """Test the SharedArray module's setitem method."""
    print()
    cache = SharedArray(
        shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
    )
    cache[index] = torch.randn(*shape[1:])
    assert cache[index] is not None
    cache.clear(index)
    assert cache[index] is None


def test_SharedArray_clear_all(
    shape=(100, 1, 64, 64),
    dtype=torch.float32,
    cache_size=1.0,
):
    """Test the SharedArray module's setitem method."""
    print()
    cache = SharedArray(
        shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
    )
    cache[0] = torch.randn(*shape[1:])
    cache[shape[0] // 2] = torch.randn(*shape[1:])
    assert cache[0] is not None
    assert cache[shape[0] // 2] is not None
    cache.clear()
    assert cache[0] is None
    assert cache[shape[0] // 2] is None


def test_SharedArray_str(
    shape=(2000, 1, 128, 128),
    dtype=torch.float32,
    cache_size=0.1,
    setindex=10,
):
    """Test the SharedArray module's str method."""
    cache = SharedArray(
        shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
    )
    cache[setindex] = torch.randn(*shape[1:])
    print(cache)


@pytest.mark.parametrize(
    "descr,cache_size",
    [
        ("metadata_cache_test", 16),
        ("metadata_cache_test", 32),
    ],
)
def test_SharedDict_init(descr, cache_size):
    """Test the SharedArray module."""
    cache_dict = SharedDict(descr=descr, size=cache_size, verbose=True)
    assert hasattr(cache_dict, "shm")
    assert cache_dict.cache_size == nbytes(f"{cache_size}M")
    cache_dict.clear_allocation()


@pytest.mark.parametrize(
    "descr,cache_size",
    [
        ("metadata_cache_test", 16),
        ("metadata_cache_test", 32),
    ],
)
def test_SharedDict_write(descr, cache_size):
    """Test the SharedArray module."""
    sample_dict = {
        "numbers": [1, 2, 3, 4, 5],
        "tensor": torch.Tensor([42, 42, 42]),
        "index": 1,
        "bool": False,
        "ratio": 0.8,
        "foo": "bar",
    }
    cache_dict = SharedDict(descr=descr, size=cache_size, verbose=True)
    cache_dict.write_buffer(sample_dict)
    dct = cache_dict.read_buffer()
    assert dct["numbers"] == sample_dict["numbers"]
    assert sample_dict["tensor"].equal(dct["tensor"])
    assert dct["index"] == sample_dict["index"]
    assert dct["bool"] == sample_dict["bool"]
    assert dct["ratio"] == sample_dict["ratio"]
    assert dct["foo"] == sample_dict["foo"]
    cache_dict.clear_allocation()


@pytest.mark.parametrize(
    "descr,cache_size",
    [
        ("metadata_cache_test", "1b"),
        ("metadata_cache_test", "5b"),
    ],
)
def test_SharedDict_buffer_too_small(descr, cache_size):
    """Test the SharedArray module."""
    sample_dict = {
        "numbers": [1, 2, 3, 4, 5],
        "index": 1,
        "bool": False,
        "ratio": 0.8,
        "foo": "bar",
    }
    with pytest.raises(ValueError):
        cache_dict = SharedDict(descr=descr, size=cache_size, verbose=True)
        cache_dict.clear_allocation()


@pytest.mark.parametrize(
    "descr,cache_size",
    [
        ("metadata_cache_test", "6b"),
        ("metadata_cache_test", "10b"),
    ],
)
def test_SharedDict_write_too_big(descr, cache_size):
    """Test the SharedArray module."""
    sample_dict = {
        "numbers": [1, 2, 3, 4, 5],
        "index": 1,
        "bool": False,
        "ratio": 0.8,
        "foo": "bar",
    }
    cache_dict = SharedDict(descr=descr, size=cache_size, verbose=True)
    data = cache_dict.write_buffer(sample_dict)
    assert data is None
    cache_dict.clear_allocation()


@pytest.mark.parametrize(
    "descr,cache_size",
    [
        ("metadata_cache_test", "10M"),
    ],
)
def test_SharedDict_open_buffer_context(descr, cache_size):
    """Test the SharedArray module."""
    sample_dict = {
        "numbers": [1, 2, 3, 4, 5],
        "index": 1,
        "bool": False,
        "ratio": 0.8,
        "foo": "bar",
    }
    cache_dict = SharedDict(descr=descr, size=cache_size, verbose=True)
    cache_dict.write_buffer(sample_dict)
    with cache_dict.open_buffer() as dct:
        assert dct == sample_dict
        dct["new"] = "entry"
    new_dct = cache_dict.read_buffer()
    assert "new" in new_dct
    cache_dict.clear_allocation()


@pytest.mark.parametrize(
    "n,descr,slot_size,cache_size",
    [
        (120, "metadata_cache_test", "150b", "16M"),
        (120, "metadata_cache_test", "650b", "16M"),
        (120, "metadata_cache_test", "850b", "16M"),
    ],
)
def test_SharedDictList_init(n, descr, slot_size, cache_size):
    """Test the SharedArray module."""

    def gen_data(n):
        return {
            "numbers": np.random.randn(4).tolist(),
            "index": n,
            "bool": False,
            "ratio": np.random.rand(1)[0],
            "foo": "bar",
        }

    print(serial_byte_size(gen_data(2)) * 120)
    meta_cache = SharedDictList(
        n,
        gen_data(1),
        gen_data(2),
        gen_data(3),
        descr=descr,
        slot_size=slot_size,
        size=cache_size,
        verbose=True,
    )
    assert isinstance(meta_cache, SharedDictList)
    assert hasattr(meta_cache, "_slots")
    assert hasattr(meta_cache, "_shm_states")
    assert len(meta_cache._slots) == len(meta_cache._shm_states)
    meta_cache.clear_allocation()


@pytest.mark.parametrize(
    "n,descr,slot_size,cache_size",
    [
        (12000, "metadata_cache_test", "650b", "16M"),
        (12000, "metadata_cache_test", "850b", "16M"),
    ],
)
def test_SharedDictList_init_no_seq(n, descr, slot_size, cache_size):
    """Test the SharedArray module."""

    def gen_data(n):
        return {
            "numbers": np.random.randn(4).tolist(),
            "index": n,
            "bool": False,
            "ratio": np.random.rand(1)[0],
            "foo": "bar",
        }

    meta_cache = SharedDictList(
        n, descr=descr, slot_size=slot_size, size=cache_size, verbose=True
    )
    assert isinstance(meta_cache, SharedDictList)
    assert hasattr(meta_cache, "_slots")
    assert hasattr(meta_cache, "_shm_states")
    assert len(meta_cache._slots) == len(meta_cache._shm_states)
    meta_cache.clear_allocation()


@pytest.mark.parametrize(
    "n,descr,slot_size,cache_size",
    [
        (12000, "metadata_cache_test", "650b", "16K"),
        (12000, "metadata_cache_test", "850b", "16K"),
    ],
)
def test_SharedDictList_init_smaller_cache(n, descr, slot_size, cache_size):
    """Test the SharedArray module."""

    def gen_data(n):
        return {
            "numbers": np.random.randn(4).tolist(),
            "index": n,
            "bool": False,
            "ratio": np.random.rand(1)[0],
            "foo": "bar",
        }

    meta_cache = SharedDictList(
        n, descr=descr, slot_size=slot_size, size=cache_size, verbose=True
    )
    assert isinstance(meta_cache, SharedDictList)
    assert hasattr(meta_cache, "_slots")
    assert hasattr(meta_cache, "_shm_states")
    assert len(meta_cache._slots) < len(meta_cache._shm_states)
    meta_cache.clear_allocation()


@pytest.mark.parametrize(
    "n,descr,slot_size,cache_size",
    [
        (12000, "metadata_cache_test", "650b", "16K"),
        (12000, "metadata_cache_test", "850b", "16K"),
    ],
)
def test_SharedDictList_setitem_and_getitem(n, descr, slot_size, cache_size):
    """Test the SharedArray module."""

    def gen_data(n):
        return {
            "numbers": np.random.randn(4).tolist(),
            "index": n,
            "bool": False,
            "ratio": np.random.rand(1)[0],
            "foo": "bar",
        }

    meta_cache = SharedDictList(
        n, descr=descr, slot_size=slot_size, size=cache_size, verbose=True
    )
    data = gen_data(0)
    meta_cache[0] = data
    cached_data = meta_cache[0]
    assert data == cached_data
    assert meta_cache.get_state(0)[0].value == 1
    assert meta_cache.get_state(1)[0].value == 0
    assert meta_cache.get_state(3)[0].value == 0
    meta_cache.clear_allocation()

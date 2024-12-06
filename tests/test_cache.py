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
import torch
from chuchichaestli.data.cache import get_max_ram, get_max_shm, SharedArray, nbytes


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

@pytest.mark.parametrize(
    "x", ["2.0G", "2.0GB", "2.0 GB", 2147483648.0, 2147483648]
)
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

@pytest.mark.parametrize(
    "x", [None, 0.0, 0, "GB"]
)
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
    

@pytest.mark.parametrize(
    "shape,dtype,cache_size",
    [
        ((100, 1, 64, 64), torch.float32, 1.0),
        ((200, 1, 32, 32), torch.float64, "4G"),
        ((300, 1, 32, 32), torch.int8, 4),
    ]
)
def test_SharedArray_init(
    shape,
    dtype,
    cache_size
):
    """Test the SharedArray module."""
    cache = SharedArray(shape, size=cache_size, dtype=dtype, verbose=True)
    assert isinstance(cache.array, torch.Tensor)
    assert isinstance(cache.states, torch.Tensor)
    assert len(cache) == shape[0]
    assert cache.get_state(0).value == 0


@pytest.mark.parametrize(
    "shape,dtype,cache_size",
    [
        ((300, 1, 32, 32), torch.int16, "40KB"),
        ((200, 1, 32, 32), torch.float32, "4B"),
    ]
)
def test_SharedArray_ooc(
    shape,
    dtype,
    cache_size
):
    """Test the SharedArray module with out-of-cache values."""
    print()
    cache = SharedArray(shape, size=cache_size, dtype=dtype, verbose=True)
    assert isinstance(cache.array, torch.Tensor)
    assert isinstance(cache.states, torch.Tensor)
    assert len(cache) < shape[0]
    assert cache.get_state(-1).value == 2


@pytest.mark.parametrize(
    "shape,dtype,cache_size,setindex,getindex",
    [
        ((100, 1, 64, 64), torch.float32, 1.0, 10, 10),
        ((100, 1, 64, 64), torch.float32, 1.0, 10, 12),
    ]
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
    cache = SharedArray(shape, size=cache_size, dtype=dtype, verbose=True)
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
    cache = SharedArray(shape, size=cache_size, dtype=dtype, verbose=True)
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
    cache = SharedArray(shape, size=cache_size, dtype=dtype, verbose=True)
    cache[0] = torch.randn(*shape[1:])
    cache[shape[0]//2] = torch.randn(*shape[1:])
    assert cache[0] is not None
    assert cache[shape[0]//2] is not None
    cache.clear()
    assert cache[0] is None
    assert cache[shape[0]//2] is None

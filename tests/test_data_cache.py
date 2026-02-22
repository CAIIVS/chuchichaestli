# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the cache module."""

import pytest
import numpy as np
import torch
from chuchichaestli.data.cache import (
    npy_to_torch_dtype,
    serial_byte_size,
    nbytes,
    SlotState,
    SharedArray,
    SharedDict,
    SharedDictList,
)


def _make_array(shape=(100, 4), dtype=torch.float32, size="4M") -> SharedArray:
    return SharedArray(shape, size=size, dtype=dtype, use_lock=False)


def _make_dict_list(n=50, slot_size="256b", size="4M") -> SharedDictList:
    return SharedDictList(n, slot_size=slot_size, size=size, use_lock=False)


def _sample_dict():
    return {"x": 1, "y": [1.0, 2.0], "flag": True}


class TestNpyToTorchDtype:
    """Unit tests for npy_to_torch_dtype."""

    @pytest.mark.parametrize(
        "np_dtype,expected",
        [
            ("bool", torch.bool),
            ("uint8", torch.uint8),
            ("int8", torch.int8),
            ("int16", torch.int16),
            ("int32", torch.int32),
            ("int64", torch.int64),
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("complex64", torch.complex64),
            ("complex128", torch.complex128),
        ],
    )
    def test_known_dtypes(self, np_dtype, expected):
        """Every supported numpy dtype maps to its torch counterpart."""
        assert npy_to_torch_dtype(np_dtype) == expected

    @pytest.mark.parametrize(
        "np_dtype,expected",
        [
            (np.dtype("float32"), torch.float32),
            (np.float32, torch.float32),
            (np.int64, torch.int64),
        ],
    )
    def test_numpy_dtype_objects(self, np_dtype, expected):
        """Accepts np.dtype objects and numpy type classes, not just strings."""
        assert npy_to_torch_dtype(np_dtype) == expected

    def test_unknown_dtype_returns_none(self):
        """An unrecognised dtype string returns None instead of raising."""
        assert npy_to_torch_dtype("float128") is None

    def test_invalid_string_returns_none(self):
        """A nonsense string that cannot be parsed returns None."""
        assert npy_to_torch_dtype("definitely_not_a_dtype") is None


class TestNbytes:
    """Unit tests for nbytes."""

    @pytest.mark.parametrize("x", ["2.0G", "2.0GB", "2.0 GB", 2147483648.0, 2147483648])
    def test_nbytes_2G(self, x):
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
    def test_nbytes_null(self, x):
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

    def test_invalid_unit_raises(self):
        """An unknown unit suffix in the string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown unit"):
            nbytes("4X")

    def test_repr(self):
        """__repr__ returns the same string as __str__ / as_str."""
        b = nbytes("1M")
        assert repr(b) == b.as_bstr()

    def test_arithmetic_preserves_type(self):
        """Class inherits float arithmetic; results are plain floats."""
        b = nbytes("1M")
        assert b + b == float(b) * 2

    def test_nbytes_size_constructor(self):
        """Passing an nbytes instance directly round-trips correctly."""
        original = nbytes("512K")
        copy = nbytes(original)
        assert copy == original
        assert isinstance(copy, nbytes)


class TestSerialByteSize:
    """Unit tests for serial_byte_size."""

    def test_serial_byte_size(self):
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

    @pytest.mark.parametrize("x", ["2.0G", "2.0GB", "2.0 GB", 2147483648.0, 2147483648])
    def test_double_nbytes(self, x):
        """Test the nbytes class double use."""
        b = nbytes(x)
        db = nbytes(b)
        assert 0 < db < 10**10
        assert isinstance(db, nbytes)
        assert isinstance(db, float)
        assert db == float(db)
        assert isinstance(db.as_str(), str)
        assert isinstance(db.as_bstr(), str)
        assert isinstance(db.to("G"), nbytes)
        assert isinstance(db.to("G"), float)

    def test_serial_byte_size_empty(self):
        """Test the serial_byte_size function."""
        dct_size = serial_byte_size({})
        assert dct_size == 5

    def test_tensor_is_serialisable(self):
        """Plain torch tensors (not tied to C-extensions) are picklable."""
        t = torch.tensor([1.0, 2.0, 3.0])
        size = serial_byte_size(t)
        assert size > 0

    def test_nested_dict(self):
        """Nested dicts produce a larger size estimate than flat ones."""
        flat = serial_byte_size({"a": 1})
        nested = serial_byte_size({"a": {"b": {"c": 1}}})
        assert nested > flat


class TestSharedArray:
    """Unit tests for SharedArray class."""

    @pytest.mark.parametrize(
        "shape,dtype,cache_size",
        [
            ((100, 1, 64, 64), torch.float32, 0.05),
            ((200, 1, 32, 32), torch.float64, "4G"),
            ((300, 1, 32, 32), torch.int8, 1),
        ],
    )
    def test_init(self, shape, dtype, cache_size):
        """Test the SharedArray module."""
        cache = SharedArray(
            shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
        )
        assert isinstance(cache.array, torch.Tensor)
        assert isinstance(cache.states, torch.Tensor)
        assert len(cache) == shape[0]
        assert cache.get_state(0)[0].value == 0

    def test_zero_size(self):
        """Test the boolean values of instance."""
        cache = SharedArray(
            shape=(),
            size=0,
            dtype=torch.float32,
            allow_overwrite=True,
            verbose=True,
        )
        assert not cache

    @pytest.mark.parametrize(
        "shape,dtype,cache_size",
        [
            ((300, 1, 32, 32), torch.int16, "40KB"),
            ((200, 1, 32, 32), torch.float32, "4B"),
        ],
    )
    def test_ooc(self, shape, dtype, cache_size):
        """Test the SharedArray module with out-of-cache values."""
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
    def test_setitem_and_getitem(
        self,
        shape,
        dtype,
        cache_size,
        setindex,
        getindex,
    ):
        """Test the SharedArray module's setitem method."""
        cache = SharedArray(
            shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
        )
        cache[setindex] = torch.randn(*shape[1:])
        if getindex == setindex:
            assert cache[getindex] is not None
            assert isinstance(cache[getindex], torch.Tensor)
        else:
            assert cache[getindex] is None

    def test_clear_index(
        self,
        shape=(100, 1, 64, 64),
        dtype=torch.float32,
        cache_size=1.0,
        index=10,
    ):
        """Test the SharedArray module's setitem method."""
        cache = SharedArray(
            shape, size=cache_size, dtype=dtype, allow_overwrite=True, verbose=True
        )
        cache[index] = torch.randn(*shape[1:])
        assert cache[index] is not None
        cache.clear(index)
        assert cache[index] is None

    def test_clear_all(
        self,
        shape=(100, 1, 64, 64),
        dtype=torch.float32,
        cache_size=1.0,
    ):
        """Test the SharedArray module's setitem method."""
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

    def test_str(
        self,
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

    def test_cached_states_after_writes(self):
        """cached_states counts exactly the number of filled slots."""
        cache = _make_array()
        assert cache.cached_states == 0
        cache[0] = torch.zeros(4)
        cache[1] = torch.zeros(4)
        assert cache.cached_states == 2
        cache.clear_allocation()

    def test_cached_bytes_after_write(self):
        """cached_bytes is non-zero and a multiple of sample size after a write."""
        cache = _make_array(shape=(10, 8))
        cache[0] = torch.zeros(8)
        assert cache.cached_bytes > 0
        cache.clear_allocation()

    def test_neg_operator(self):
        """__neg__ returns True for a zero-size cache."""
        cache = SharedArray(shape=(), size=0)
        assert -cache
        cache.clear_allocation()

    def test_repr(self):
        """__repr__ delegates to __str__."""
        cache = _make_array()
        assert repr(cache) == str(cache)
        cache.clear_allocation()

    def test_contains_true_after_write(self):
        """Index in cache is True after the slot has been filled."""
        cache = _make_array()
        cache[5] = torch.zeros(4)
        assert 5 in cache
        cache.clear_allocation()

    def test_contains_false_before_write(self):
        """Index in cache is False for an empty slot."""
        cache = _make_array()
        assert 5 not in cache
        cache.clear_allocation()

    def test_contains_false_for_ooc(self):
        """An OOC index is not considered 'in' the cache."""
        # Very small cache so only a few slots fit
        cache = SharedArray(shape=(100, 64, 64), size="4B", dtype=torch.float32)
        # The last state index will be OOC
        assert (cache.states == SlotState.OOC.value).any()
        ooc_idx = int((cache.states == SlotState.OOC.value).nonzero()[0].item())
        assert ooc_idx not in cache
        cache.clear_allocation()

    def test_none_index_returns_invalid(self):
        """get_state(None) returns (INVALID, None)."""
        cache = _make_array()
        state, idx = cache.get_state(None)
        assert state == SlotState.INVALID
        assert idx is None
        cache.clear_allocation()

    def test_negative_index(self):
        """A negative index wraps around correctly."""
        cache = _make_array(shape=(10, 4))
        cache[9] = torch.zeros(4)
        state, idx = cache.get_state(-1)
        assert state == SlotState.SET
        assert idx == 9
        cache.clear_allocation()

    def test_out_of_range_raises(self):
        """An index beyond dataset size raises IndexError."""
        cache = _make_array(shape=(10, 4))
        with pytest.raises(IndexError):
            cache.get_state(999)
        cache.clear_allocation()

    def test_allow_overwrite_false_raises(self):
        """Writing to an already-filled slot raises RuntimeError when locked."""
        cache = SharedArray(
            shape=(10, 4),
            size="4M",
            dtype=torch.float32,
            use_lock=False,
            allow_overwrite=False,
        )
        cache[0] = torch.zeros(4)
        with pytest.raises(RuntimeError, match="overwrites"):
            cache[0] = torch.ones(4)
        cache.clear_allocation()

    def test_shape_mismatch_raises(self):
        """Writing a tensor with the wrong shape raises ValueError."""
        cache = _make_array(shape=(10, 4))
        with pytest.raises(ValueError, match="Shape mismatch"):
            cache[0] = torch.zeros(7)  # should be (4,)
        cache.clear_allocation()

    def test_write_to_none_index_is_noop(self):
        """Writing to index=None is silently ignored."""
        cache = _make_array()
        cache[None] = torch.zeros(4)  # must not raise
        assert cache.cached_states == 0
        cache.clear_allocation()

    def test_write_to_ooc_index_is_noop(self):
        """Writing to an OOC slot does nothing."""
        cache = SharedArray(shape=(100, 64, 64), size="4B", dtype=torch.float32)
        ooc_idx = int((cache.states == SlotState.OOC.value).nonzero()[0].item())
        before = cache.cached_states
        cache[ooc_idx] = torch.zeros(64, 64)
        assert cache.cached_states == before
        cache.clear_allocation()

    def test_use_lock_false(self):
        """Option use_lock=False still writes and reads correctly."""
        cache = SharedArray(
            shape=(10, 4),
            size="4M",
            dtype=torch.float32,
            use_lock=False,
        )
        sample = torch.arange(4, dtype=torch.float32)
        cache[3] = sample
        assert torch.allclose(cache[3], sample)
        cache.clear_allocation()

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
        ],
    )
    def test_supported_dtype(self, dtype):
        """Test supported data types."""
        cache = SharedArray(shape=(10, 4), size="4M", dtype=dtype, use_lock=False)
        sample = torch.zeros(4, dtype=dtype)
        cache[0] = sample
        result = cache[0]
        assert result is not None
        assert result.dtype == dtype
        cache.clear_allocation()

    def test_unsupported_dtype_raises(self):
        """An unsupported dtype raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            SharedArray(shape=(10, 4), size="4M", dtype=torch.float16)


class TestSharedDict:
    """Unit tests for SharedDict class."""

    def test_nbytes_size_constructor(self):
        """Passing nbytes directly as size works correctly."""
        sd = SharedDict(size=nbytes("1M"), use_lock=False)
        sd["k"] = 1
        assert sd["k"] == 1
        sd.clear_allocation()

    @pytest.mark.parametrize(
        "descr,cache_size",
        [
            ("metadata_cache_test", 0.016),
            ("metadata_cache_test", 0.032),
        ],
    )
    def test_init_shm(self, descr, cache_size):
        """Test the SharedArray module."""
        cache_dict = SharedDict(descr=descr, size=cache_size, verbose=True)
        assert hasattr(cache_dict, "shm")
        assert cache_dict.cache_size == nbytes(f"{cache_size}G")
        cache_dict.clear_allocation()

    @pytest.mark.parametrize(
        "descr,cache_size",
        [
            ("metadata_cache_test", 16),
            ("metadata_cache_test", 32),
        ],
    )
    def test_write_buffer(self, descr, cache_size):
        """Test the SharedDict write_buffer method."""
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
    def test_write_buffer_too_small(self, descr, cache_size):
        """Test the SharedDict write_buffer method when samples are too small."""
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
    def test_write_buffer_too_big(self, descr, cache_size):
        """Test the SharedDict write_buffer method when samples are too big."""
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
    def test_open_buffer_context(self, descr, cache_size):
        """Test the SharedDict module."""
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

    def test_kwargs_preload(self):
        """Key-value kwargs passed at construction are immediately available."""
        sd = SharedDict(size="1M", use_lock=False, foo="bar", count=7)
        assert sd["foo"] == "bar"
        assert sd["count"] == 7
        sd.clear_allocation()

    def test_allow_overwrite_false_raises(self):
        """Writing to an existing key raises RuntimeError when locked."""
        sd = SharedDict(size="1M", use_lock=False, allow_overwrite=False)
        with pytest.raises(RuntimeError, match="overwrites"):
            sd["x"] = 2
        sd.clear_allocation()

    def test_use_lock_false(self):
        """use_lock=False does not affect correctness for single-threaded use."""
        sd = SharedDict(size="1M", use_lock=False)
        sd["k"] = 42
        assert sd["k"] == 42
        sd.clear_allocation()

    @pytest.fixture
    def sd_ab(self):
        """Simple instance fixture with a and b keys."""
        d = SharedDict(size="1M", use_lock=False)
        d["a"] = 1
        d["b"] = [1, 2, 3]
        yield d
        d.clear_allocation()

    def test_getitem(self, sd_ab):
        """Test getter method."""
        assert sd_ab["a"] == 1

    def test_setitem_and_getitem(self):
        """Test setter and getter methods."""
        sd = SharedDict(size="1M", use_lock=False)
        sd["key"] = 42
        assert sd["key"] == 42
        sd.clear_allocation()

    def test_delitem(self, sd_ab):
        """Test instance deletion."""
        del sd_ab["a"]
        assert "a" not in sd_ab

    def test_len(self, sd_ab):
        """Test instance length."""
        assert len(sd_ab) == 2

    def test_iter(self, sd_ab):
        """Test iterator method."""
        keys = list(sd_ab)
        assert set(keys) == {"a", "b"}

    def test_contains_true(self, sd_ab):
        """Test in trait."""
        assert "a" in sd_ab

    def test_contains_false(self, sd_ab):
        """Test not in trait."""
        assert "missing" not in sd_ab

    def test_eq(self, sd_ab):
        """Test equality."""
        assert sd_ab == {"a": 1, "b": [1, 2, 3]}

    def test_ne(self, sd_ab):
        """Test negation."""
        assert sd_ab != {"a": 99}

    def test_or(self, sd_ab):
        """Test or."""
        merged = sd_ab | {"c": 3}
        assert merged["c"] == 3

    def test_ror(self, sd_ab):
        """Test reversed or."""
        merged = {"c": 3} | sd_ab
        assert merged["a"] == 1
        assert merged["c"] == 3

    def test_str(self, sd_ab):
        """Test str."""
        assert "@shm" in str(sd_ab)

    def test_repr(self, sd_ab):
        """Test repr."""
        assert "@shm" in repr(sd_ab)

    @pytest.fixture
    def sd_xy(self):
        """Simple instance fixture with x and y keys."""
        d = SharedDict(size="1M", use_lock=False)
        d["x"] = 10
        d["y"] = 20
        yield d
        d.clear_allocation()

    def test_get_existing(self, sd_xy):
        """Test get method."""
        assert sd_xy.get("x") == 10

    def test_get_missing_with_default(self, sd_xy):
        """Test get method's fallback."""
        assert sd_xy.get("missing", -1) == -1

    def test_keys(self, sd_xy):
        """Test keys method."""
        assert set(sd_xy.keys()) == {"x", "y"}

    def test_values(self, sd_xy):
        """Test values method."""
        assert set(sd_xy.values()) == {10, 20}

    def test_items(self, sd_xy):
        """Test items method."""
        assert dict(sd_xy.items()) == {"x": 10, "y": 20}

    def test_update_dict(self, sd_xy):
        """Test update method."""
        sd_xy.update({"z": 30})
        assert sd_xy["z"] == 30

    def test_update_kwargs(self, sd_xy):
        """Test update method with alternative input format."""
        sd_xy.update(w=40)
        assert sd_xy["w"] == 40

    def test_setdefault_missing_key(self, sd_xy):
        """Test setdefault method's fallback."""
        val = sd_xy.setdefault("new_key", 99)
        assert val == 99
        assert sd_xy["new_key"] == 99

    def test_setdefault_existing_key(self, sd_xy):
        """Test setdefault method."""
        val = sd_xy.setdefault("x", 999)
        assert val == 10  # should not overwrite

    def test_pop_existing(self, sd_xy):
        """Test pop method."""
        val = sd_xy.pop("x")
        assert val == 10
        assert "x" not in sd_xy

    def test_pop_missing_with_default(self, sd_xy):
        """Tes pop method's fallback."""
        val = sd_xy.pop("missing", "fallback")
        assert val == "fallback"

    def test_pop_missing_raises(self, sd_xy):
        """Test pop method's key error."""
        with pytest.raises(KeyError):
            sd_xy.pop("definitely_not_present")

    def test_clear(self, sd_xy):
        """Test clear method."""
        sd_xy.clear()
        assert len(sd_xy) == 0


class TestSharedDictList:
    """Unit tests for SharedDictList class."""

    @pytest.mark.parametrize(
        "n,descr,slot_size,cache_size",
        [
            (120, "metadata_cache_test", "150b", "16M"),
            (120, "metadata_cache_test", "650b", "16M"),
            (120, "metadata_cache_test", "850b", "16M"),
        ],
    )
    def test_init(self, n, descr, slot_size, cache_size):
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
    def test_init_no_seq(self, n, descr, slot_size, cache_size):
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
    def test_init_smaller_cache(self, n, descr, slot_size, cache_size):
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
    def test_setitem_and_getitem(self, n, descr, slot_size, cache_size):
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

    def test_zero_size(self):
        """Test the boolean values of instance."""
        meta_cache = SharedDictList(0, size=0, verbose=True)
        print(meta_cache)
        assert not meta_cache
        meta_cache.clear_allocation()

    def test_cached_states_after_writes(self):
        """Test cache states after caching."""
        cache = _make_dict_list()
        assert cache.cached_states == 0
        cache[0] = _sample_dict()
        cache[1] = _sample_dict()
        assert cache.cached_states == 2
        cache.clear_allocation()

    def test_cached_bytes_after_write(self):
        """Test cached bytes."""
        cache = _make_dict_list()
        cache[0] = _sample_dict()
        assert cache.cached_bytes > 0
        cache.clear_allocation()

    def test_neg_operator(self):
        """__neg__ returns True for a zero-size cache."""
        cache = SharedDictList(0, size=0)
        assert -cache
        cache.clear_allocation()

    def test_repr(self):
        """Test repr method."""
        cache = _make_dict_list()
        assert repr(cache) == str(cache)
        cache.clear_allocation()

    def test_contains_true_after_write(self):
        """Test item contained case."""
        cache = _make_dict_list()
        cache[3] = _sample_dict()
        assert 3 in cache
        cache.clear_allocation()

    def test_contains_false_before_write(self):
        """Test item not contained case."""
        cache = _make_dict_list()
        assert 3 not in cache
        cache.clear_allocation()

    def test_contains_false_for_ooc(self):
        """An OOC index reports False for __contains__."""
        cache = SharedDictList(100, slot_size="256b", size="64b")
        ooc_idx = int((cache.states == SlotState.OOC.value).nonzero()[0].item())
        assert ooc_idx not in cache
        cache.clear_allocation()

    def test_none_index_returns_invalid(self):
        """Test invalid index state."""
        cache = _make_dict_list()
        state, idx = cache.get_state(None)
        assert state == SlotState.INVALID
        assert idx is None
        cache.clear_allocation()

    def test_negative_index(self):
        """Test negative index case."""
        cache = _make_dict_list(n=10)
        cache[9] = _sample_dict()
        state, idx = cache.get_state(-1)
        assert state == SlotState.SET
        assert idx == 9
        cache.clear_allocation()

    def test_out_of_range_raises(self):
        """Test index out of range case."""
        cache = _make_dict_list(n=10)
        with pytest.raises(IndexError):
            cache.get_state(999)
        cache.clear_allocation()

    def test_allow_overwrite_false_raises(self):
        """Test not allowing overwrites."""
        cache = SharedDictList(
            50,
            slot_size="256b",
            size="4M",
            use_lock=False,
            allow_overwrite=False,
        )
        cache[0] = _sample_dict()
        with pytest.raises(RuntimeError, match="overwrites"):
            cache[0] = _sample_dict()
        cache.clear_allocation()

    def test_dict_exceeds_slot_raises(self):
        """Writing a dict larger than the slot size raises RuntimeError."""
        cache = SharedDictList(50, slot_size="8b", size="4M", use_lock=False)
        big_dict = {"data": list(range(500))}
        with pytest.raises(RuntimeError, match="exceeds slot size"):
            cache[0] = big_dict
        cache.clear_allocation()

    def test_write_to_ooc_is_noop(self):
        """Writing to an OOC slot is silently ignored."""
        cache = SharedDictList(100, slot_size="256b", size="64b")
        ooc_idx = int((cache.states == SlotState.OOC.value).nonzero()[0].item())
        before = cache.cached_states
        cache[ooc_idx] = _sample_dict()
        assert cache.cached_states == before
        cache.clear_allocation()

    def test_use_lock_false(self):
        """Option use_lock=False still writes and reads correctly."""
        cache = SharedDictList(50, slot_size="256b", size="4M", use_lock=False)
        data = _sample_dict()
        cache[5] = data
        assert cache[5] == data
        cache.clear_allocation()

    def test_clear_single_index(self):
        """Test clearing a single index."""
        cache = _make_dict_list()
        cache[0] = _sample_dict()
        cache[1] = _sample_dict()
        assert cache.cached_states == 2
        cache.clear(0)
        assert cache.cached_states == 1
        assert cache[0] is None
        assert cache[1] is not None
        cache.clear_allocation()

    def test_clear_all(self):
        """Test clearing all cached entries."""
        cache = _make_dict_list()
        for i in range(5):
            cache[i] = _sample_dict()
        assert cache.cached_states == 5
        cache.clear()
        assert cache.cached_states == 0
        cache.clear_allocation()

    def test_clear_none_clears_all(self):
        """clear(None) is equivalent to clear()."""
        cache = _make_dict_list()
        cache[0] = _sample_dict()
        cache.clear(None)
        assert cache.cached_states == 0
        cache.clear_allocation()

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the small tensor helpers."""

import numpy as np
import pytest
import torch
from chuchichaestli.utils.tensors import (
    as_array,
    npy_to_torch_dtype,
    torch_to_npy_dtype,
)


class TestAsArray:
    """Unit tests for as_array."""

    def test_shares_a_host_tensor_memory(self):
        """A tensor on the host is handed over as a view, not a copy."""
        tensor = torch.arange(6.0)
        assert np.shares_memory(as_array(tensor), tensor.numpy())

    def test_detaches_from_the_graph(self):
        """A tensor that requires grad converts instead of raising."""
        assert as_array(torch.zeros(3, requires_grad=True)).shape == (3,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a gpu")
    def test_brings_a_device_tensor_home(self):
        """A tensor on a gpu converts instead of raising."""
        assert as_array(torch.zeros(3, device="cuda")).shape == (3,)

    def test_keeps_the_dtype(self):
        """The array holds what the tensor held, not a promoted copy."""
        assert as_array(torch.zeros(2, dtype=torch.float64)).dtype == np.float64
        assert as_array(torch.zeros(2, dtype=torch.uint8)).dtype == np.uint8

    def test_passes_an_array_through(self):
        """An array is already one, so it is not copied."""
        array = np.arange(3)
        assert as_array(array) is array

    def test_reads_a_sequence(self):
        """Anything numpy can read as an array is read as one."""
        assert as_array([[1, 2], [3, 4]]).shape == (2, 2)


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


class TestTorchToNpyDtype:
    """Unit tests for torch_to_npy_dtype."""

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.bool, "bool"),
            (torch.uint8, "uint8"),
            (torch.int32, "int32"),
            (torch.int64, "int64"),
            (torch.float16, "float16"),
            (torch.float32, "float32"),
            (torch.float64, "float64"),
            (torch.complex64, "complex64"),
        ],
    )
    def test_known_dtypes(self, dtype, expected):
        """Every torch dtype numpy shares maps to its numpy counterpart."""
        assert torch_to_npy_dtype(dtype) == np.dtype(expected)

    def test_returns_a_dtype_instance(self):
        """The result is an `np.dtype`, ready for `np.ndarray` and friends."""
        assert isinstance(torch_to_npy_dtype(torch.float32), np.dtype)

    @pytest.mark.parametrize(
        "dtype",
        [torch.bool, torch.uint8, torch.int16, torch.int64, torch.float32,
         torch.float64, torch.complex128],
    )
    def test_round_trips(self, dtype):
        """The two directions are inverses wherever both dtypes exist."""
        assert npy_to_torch_dtype(torch_to_npy_dtype(dtype)) == dtype

    def test_element_size_is_preserved(self):
        """A translated dtype is as wide as the one it came from."""
        for dtype in (torch.uint8, torch.float32, torch.float64):
            width = torch.empty((), dtype=dtype).element_size()
            assert torch_to_npy_dtype(dtype).itemsize == width

    def test_unsupported_dtype_raises(self):
        """A dtype numpy has no counterpart for fails loudly, not silently."""
        with pytest.raises(TypeError):
            torch_to_npy_dtype(torch.bfloat16)

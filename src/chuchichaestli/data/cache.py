"""Caching for tensors from PyTorch datasets.

Inspired by: https://github.com/ptrblck/pytorch_misc/blob/master/shared_array.py

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

import psutil
import pickle
from enum import Enum
import ctypes
from functools import wraps
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory, ShareableList
import numpy as np
import torch
from typing import Protocol, Any
from collections.abc import Generator, Iterator, KeysView, ValuesView, ItemsView


__all__ = [
    "SharedArray",
    "SharedDict",
    "SharedDictList",
    "nbytes",
    "get_max_ram",
    "get_max_shm",
    "serial_byte_size",
]


BYTE_UNITS = {
    "b": 1,
    "K": 1 << 10,
    "M": 1 << 20,
    "G": 1 << 30,
    "T": 1 << 40,
    "P": 1 << 50,
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "PB": 10**15,
}

C_DTYPES = {
    torch.bool: ctypes.c_bool,
    torch.uint8: ctypes.c_uint8,
    torch.int8: ctypes.c_int8,
    torch.int16: ctypes.c_int16,
    torch.int32: ctypes.c_int32,
    torch.int64: ctypes.c_int64,
    torch.float32: ctypes.c_float,
    torch.float64: ctypes.c_double,
}

NPY_TO_TORCH_DTYPES = {
    np.bool: torch.bool,
    "bool": torch.bool,
    np.uint8: torch.uint8,
    "uint8": torch.uint8,
    np.int8: torch.int8,
    "int8": torch.int8,
    np.int16: torch.int16,
    "int16": torch.int16,
    np.int32: torch.int32,
    "int32": torch.int32,
    np.int64: torch.int64,
    "int64": torch.int64,
    np.float16: torch.float16,
    "float16": torch.float16,
    np.float32: torch.float32,
    "float32": torch.float32,
    np.float64: torch.float64,
    "float64": torch.float64,
    np.complex64: torch.complex64,
    "complex64": torch.complex64,
    np.complex128: torch.complex128,
    "complex128": torch.complex128,
}


def npy_to_torch_dtype(dtype: str | np.dtype) -> torch.dtype:
    """Converts numpy to torch data types."""
    dtype = str(dtype)
    return (
        NPY_TO_TORCH_DTYPES[str(dtype)] if dtype in NPY_TO_TORCH_DTYPES.keys() else None
    )


class nbytes(float):
    """A float class which accepts byte size strings, e.g. '4.2 GB'."""

    __slots__ = ["units"]

    def __new__(cls, n_bytes: int | float | str | None = None) -> "nbytes":
        """Translate a byte size string into a proper integer.

        Args:
          n_bytes: An integer (in bytes), float (in bytes), or byte string,
            i.e. '1K'=1024, or '1KB'=1000.
        """
        cls.units = BYTE_UNITS
        if n_bytes is None:
            n_bytes = 0
        elif isinstance(n_bytes, str):
            unit = "".join(i for i in n_bytes if not (i.isdigit() or i in ["."]))
            unit = unit.strip()
            units = cls.units[unit]
            n_bytes = n_bytes.replace(unit, "").strip()
            if not n_bytes:
                n_bytes = 0
            n_bytes = float(n_bytes)
            n_bytes = n_bytes * units
        return float.__new__(cls, n_bytes)

    def __add__(self, other: int | float) -> "nbytes":
        """Addition of nbyte instances."""
        return self.__class__(float.__add__(self, float(other)))

    def __radd__(self, other: int | float) -> "nbytes":
        """Addition of nbyte instances."""
        return self.__class__(float.__radd__(self, float(other)))

    def __mul__(self, other: int | float) -> "nbytes":
        """Multiplication of nbyte instances."""
        return self.__class__(float.__mul__(self, float(other)))

    def __rmul__(self, other: int | float) -> "nbytes":
        """Multiplication of nbyte instances."""
        return self.__class__(float.__rmul__(self, float(other)))

    def __truediv__(self, other: int | float) -> "nbytes":
        """Division (true) of nbyte instances."""
        return self.__class__(float.__truediv__(self, float(other)))

    def __floordiv__(self, other: int | float) -> "nbytes":
        """Division (floor) of nbyte instances."""
        return self.__class__(float.__floordiv__(self, float(other)))

    def __str__(self) -> str:
        """String of instance."""
        return self.as_bstr()

    def __repr__(self) -> str:
        """Representation of instance."""
        return self.as_bstr()

    def as_str(self) -> str:
        """Parse to string in decimal units."""
        for k in self.units:
            if not k.endswith("B"):
                continue
            if 1 <= int(self) // self.units[k] < 999:
                return f"{self / self.units[k]:.2f}{k}"
        return "0B"

    def as_bstr(self) -> str:
        """Parse to string in binary units."""
        for k in self.units:
            if not k.endswith("B") and 1 <= int(self) // self.units[k] < 999:
                return f"{self / self.units[k]:.2f}{k}"
        return "0B"

    def to(self, unit: str) -> str:
        """Convert to unit."""
        if unit in self.units:
            return self.__class__(self / self.units[unit])
        else:
            raise ValueError(f"Unknown unit, choose from {list(self.units.keys())}.")


class DictSerializer(Protocol):
    def dumps(self, dct: dict) -> bytes: ...

    def loads(self, data: bytes) -> dict: ...


class PickleSerializer:
    def dumps(self, dct: dict) -> bytes:
        return pickle.dumps(dct, pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> dict:
        return pickle.loads(data)


class DummyLock:
    def acquire(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass


def lock(fn):
    """Multiprocessing lock decorator.

    Note: This requires the instance to have a _lock attribute.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self._lock.acquire()
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._lock.release()

    return wrapper


def get_max_ram() -> nbytes:
    """Get the maximal size of total RAM."""
    return nbytes(psutil.virtual_memory().total)


def get_max_shm() -> nbytes:
    """Get the maximal size of available shared memory."""
    return nbytes(psutil.Process().memory_info().rss)


def serial_byte_size(
    dct: Any,
    serializer: type[DictSerializer] = PickleSerializer,
) -> nbytes:
    """Estimate the size of a serializable object (e.g. a dictionary).

    Args:
      dct: Any (through pickle) serializable data (e.g. a dictionary)
      serializer: Callable for the encoding of the input data (e.g. dictionary).
    """
    enc_bstr = serializer().dumps(dct)
    return nbytes(len(enc_bstr))


class SlotState(Enum):
    INVALID = -1
    EMPTY = 0
    SET = 1
    OOC = 2  # not in cache but valid dataset index


class SharedArray:
    """A shared memory array for use as tensor data cache in PyTorch datasets.

    Uses a ctypes multiprocessing array on shared memory as basis.
    If the size of the dataset exceeds the size of the set cache limit,
    only the first N samples will be cached. For shuffled datasets,
    this is called 'stochastic caching'.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        size: int | float | str = "4.0G",
        dtype: torch.dtype = torch.float32,
        use_lock: bool = True,
        allow_overwrite: bool = True,
        verbose: bool = False,
    ):
        """Constructor.

        Args:
            shape: Dataset N-d shape, e.g. (n_samples, channels, width, height, depth).
            size: Maximum cache size in GiB (if int or float); default "4.0 GiB".
            dtype: PyTorch dtype (default torch.float32). Must be type with corresponding ctype,
              i.e. bool, uint8, int8/16/32/64, or float32/64.
            use_lock: If True, applies a threading lock for multiprocessing.
            allow_overwrite: If True, cache slots can be overwritten.
            verbose: Print information to the stdout.
        """
        self.cache_size = (
            nbytes(f"{size}G") if isinstance(size, int | float) else nbytes(size)
        )
        self.allow_overwrite = allow_overwrite
        self.verbose = verbose

        if dtype not in C_DTYPES:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be one of {C_DTYPES.keys()}"
            )
        slot_size = int(np.prod(shape[1:]))
        slot_bytes = nbytes(slot_size * nbytes(dtype.itemsize))
        dataset_bytes = nbytes(shape[0] * slot_bytes)
        cache_bytes = self.cache_size.to("B")
        states_bytes = nbytes(shape[0] * torch.uint8.itemsize)

        total_bytes = dataset_bytes + states_bytes

        if total_bytes > cache_bytes:
            n_slots = int((cache_bytes - states_bytes) / slot_bytes)
            if self.verbose:
                print(
                    f"Requested memory ({total_bytes}) "
                    f"exceeds cache size ({self.cache_size}).\n"
                    f"Allocating cache for {n_slots} / {shape[0]} data samples."
                )
        else:
            n_slots = shape[0]
            if self.verbose:
                print(
                    f"Data size ({total_bytes}) fits into "
                    f"requested cache size ({self.cache_size}).\n"
                    f"Allocating cache for {n_slots} data samples."
                )

        mp_arr = mp.Array(C_DTYPES[dtype], n_slots * slot_size, lock=use_lock)  # type: ignore
        shm_arr = np.ctypeslib.as_array(mp_arr.get_obj())
        shm_arr = shm_arr.reshape((n_slots, *shape[1:]))
        self._slots = torch.from_numpy(shm_arr)
        self._slots *= 0

        mp_states_arr = mp.Array(C_DTYPES[torch.uint8], shape[0])  # type: ignore
        shm_states_arr = np.ctypeslib.as_array(mp_states_arr.get_obj())
        self._shm_states = torch.from_numpy(shm_states_arr)
        self._shm_states *= 0
        self._shm_states[n_slots:] = SlotState.OOC.value

    @property
    def array(self) -> torch.Tensor:
        """Shared-memory tensor data array (cached samples)."""
        return self._slots

    @property
    def states(self) -> torch.Tensor:
        """Shared-memory tensor state array.

        Note:
          - states[index] == 0 means sample at index are not yet cached.
          - states[index] == 1 means sample at index are cached.
          - states[index] == 2 means sample at index cannot be cached (due to cache limit).
        """
        return self._shm_states

    @property
    def cached_states(self) -> int:
        """Written states in cache."""
        return int(sum([s == 1 for s in self.states]))

    @property
    def cached_bytes(self) -> "nbytes":
        """Bytes written to cache."""
        n_slots = len(self.array)
        return self.cache_size * self.cached_states / n_slots

    def get_state(self, index: int | None) -> tuple[SlotState, int | None]:
        """Get the slot state at specified index."""
        if index is None:
            return SlotState.INVALID, None
        if index < 0:
            index = len(self.states) + index
        if index < 0 or index >= len(self.states):
            raise IndexError(
                f"Index {index} out of range for dataset {list(self.states.shape)}"
            )
        return SlotState(self.states[index].item()), index

    def clear(self, index: int | None = None):
        """Clear the cache (optionally only at a specified index)."""
        _, index = self.get_state(index)
        if index is None:
            self._slots *= 0
            self._shm_states *= 0
            self._shm_states[len(self) :] = SlotState.OOC.value
        else:
            self[index] = torch.zeros(self._slots.shape[1:])
            self._shm_states[index] = torch.zeros(self._shm_states.shape[1:])
            if len(self) <= index:
                self._shm_states[index] = SlotState.OOC.value

    def clear_allocation(self):
        """Delete shared memory allocation."""
        del self._slots
        del self._shm_states

    def __getitem__(self, index: int | None) -> torch.Tensor | None:
        """Fetch sample tensor from cache if stored, otherwise returns None."""
        state, index = self.get_state(index)
        if state != SlotState.SET:
            return None
        return self.array[index]

    def __setitem__(self, index: int | None, item: torch.Tensor):
        """Fill the cache at specified location."""
        state, index = self.get_state(index)
        if state == SlotState.OOC or state == SlotState.INVALID:
            return
        if state == SlotState.SET and not self.allow_overwrite:
            raise RuntimeError(
                f"{self} is locked and does not allow overwrites at {index=}."
            )
        self.array[index] = item
        self.states[index] = SlotState.SET.value

    def __contains__(self, index: int | None) -> bool:
        """Test 'in' cache in shared memory."""
        state, index = self.get_state(index)
        return state == SlotState.SET

    def __len__(self) -> int:
        """Length of the cache (may differ from dataset length depending on cache size)."""
        return len(self.array)

    def __str__(self) -> str:
        """String of the instance."""
        n_slots = len(self.array)
        n = len(self.states)
        return f"{self.__class__.__name__}({self.cached_states}({n_slots})/{n}@{self.cache_size.as_str()})"

    def __repr__(self) -> str:
        """Representation of the instance."""
        return self.__str__()


class SharedDict:
    """A shared memory dictionary for use as metadata cache in PyTorch datasets.

    Note: Serialization of PyTorch tensors is in principle possible, but it is recommended
      to use the SharedArray class for tensor types instead.
    """

    def __init__(
        self,
        descr: str = "shm_dict",
        size: int | float | str = "16M",
        serializer: DictSerializer = PickleSerializer(),
        use_lock: bool = True,
        allow_overwrite: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """Constructor.

        Args:
            descr: Descriptor ID for shared memory access.
            size: Maximum cache size in MiB (if int or float); default "16.0 GiB".
            serializer: Serializer for the encoding of the dictionary data.
            use_lock: If True, applies a threading lock for multiprocessing.
            allow_overwrite: If True, cache slots can be overwritten.
            verbose: Print information to the stdout.
            kwargs: Key-value dictionary pairs to load into memory.

        Note: If the dictionary is supposed to contain the keys
          ['descr', 'size', 'sample_size', 'allow_overwrite', 'serlializer', 'verbose']
        """
        super().__init__()
        self.descr = descr
        self.allow_overwrite = True
        self.cache_size = (
            nbytes(f"{size}M") if isinstance(size, int | float) else nbytes(size)
        )
        if self.cache_size <= 5:
            raise ValueError("Chosen cache size is too small!")
        self.serializer = serializer
        self._lock: mp.synchronize.Lock | DummyLock
        if use_lock:
            self._lock = Lock()
        else:
            self._lock = DummyLock()
        self.verbose = verbose
        self.shm = self.get_allocation()
        self.clear()
        self.update(kwargs)
        self.allow_overwrite = allow_overwrite

    def get_allocation(
        self, name: str | None = None, size: int | float | str | None = None, **kwargs
    ) -> SharedMemory:
        """Get a shared memory allocation.

        Args:
            name: Descriptor ID for shared memory access.
            size: Number of samples in the dataset.
            kwargs: Additional keywords for compatibility.
        """
        if name is None:
            name = self.descr
        if size is None or size == 0:
            size = self.cache_size
        try:
            shm = SharedMemory(name=name)
        except FileNotFoundError:
            shm = SharedMemory(name=name, create=True, size=int(size))
        return shm

    def clear_allocation(
        self,
    ):
        """Delete shared memory allocation."""
        if hasattr(self, "shm"):
            try:
                self.shm.close()
                self.shm.unlink()
            except FileNotFoundError:
                self.shm.close()

    def write_buffer(self, data: dict) -> bytes | None:
        """Write data to the shared memory buffer.

        Returns:
            Written data if it was successfully written to the buffer, None otherwise.
        """
        byte_data = self.serializer.dumps(data)
        if not hasattr(self, "shm"):
            self.get_allocation()
        if len(byte_data) < self.cache_size:
            self.shm.buf[: len(byte_data)] = byte_data
        else:
            return None
        return byte_data

    def read_buffer(self) -> dict:
        """Read the shared memory buffer."""
        return self.serializer.loads(self.shm.buf.tobytes())

    @contextmanager
    @lock
    def open_buffer(self) -> Generator:
        """Buffer context manager."""
        dct = self.read_buffer()
        yield dct
        self.write_buffer(dct)

    @lock
    def clear(self):
        """Clear the data in the shared memory buffer."""
        self.write_buffer({})

    def __del__(self):
        """Delete dictionary in shared memory."""
        self.clear_allocation()

    def __getitem__(self, key: str | int) -> Any:
        """Get item from dictionary in shared memory."""
        return self.read_buffer()[key]

    def __setitem__(self, key: str | int, value: Any):
        """Set item of dictionary in shared memory."""
        if not self.allow_overwrite:
            return
        with self.open_buffer() as dct:
            dct[key] = value

    def __delitem__(self, key: str | int):
        """Delete item from dictionary in shared memory."""
        with self.open_buffer() as dct:
            del dct[key]

    def __len__(self):
        """Get length of dictionary in shared memory."""
        return len(self.read_buffer())

    def __iter__(self) -> Iterator:
        """Iterate through dictionary in shared memory."""
        return iter(self.read_buffer())

    def __contains__(self, key: str | int) -> bool:
        """Test 'in' dictionary in shared memory."""
        return key in self.read_buffer()

    def __eq__(self, other: Any) -> bool:
        """Test equal dictionary in shared memory."""
        return self.read_buffer() == other

    def __ne__(self, other: Any) -> bool:
        """Test not equal dictionary in shared memory."""
        return self.read_buffer() != other

    def __or__(self, other: Any) -> bool:
        """Test 'or' dictionary in shared memory."""
        return self.read_buffer() | other

    def __ror__(self, other: Any) -> bool:
        """Test 'or' dictionary in shared memory."""
        return other | self.read_buffer()

    def __str__(self) -> str:
        """String of dictionary in shared memory."""
        return str(self.read_buffer()) + "@shm"

    def __repr(self) -> str:
        """Representation of dictionary in shared memory."""
        return repr(self.read_buffer()) + "@shm"

    def get(self, key: str | int, default: Any | None = None) -> Any:
        """Getter (value by key) for dictionary in shared memory."""
        return self.read_buffer().get(key, default)

    def keys(self) -> KeysView[str | int]:
        """Getter for keys of dictionary in shared memory."""
        return self.read_buffer().keys()

    def values(self) -> ValuesView:
        """Getter for values of dictionary in shared memory."""
        return self.read_buffer().values()

    def items(self) -> ItemsView:
        """Getter for items of dictionary in shared memory."""
        return self.read_buffer().items()

    def update(self, other=(), /, **kwargs):
        """Update the dictionary in shared memory."""
        with self.open_buffer() as dct:
            dct.update(other, **kwargs)

    def setdefault(self, key: str | int, default: Any | None = None) -> dict:
        """Setdefault the dictionary in shared memory."""
        with self.open_buffer() as dct:
            return dct.setdefault(key, default)

    def pop(self, key: str | int, default: Any | None = None):
        """Pop the dictionary in shared memory."""
        with self.open_buffer() as dct:
            if default is None or default is object():
                return dct.pop(key)
            return dct.pop(key, default)


class SharedDictList:
    """A shared memory list of dictionaries for use as metadata cache in PyTorch datasets.

    Note: Serialization of PyTorch tensors is in principle possible, but it is recommended
      to use the SharedArray class for tensor types instead.
    """

    def __init__(
        self,
        n: int,
        *sequence: list[int | float | bool | str | bytes | dict | None],
        descr: str = "shm_list",
        slot_size: int | float | str = "650b",
        size: int | float | str = "64M",
        serializer: DictSerializer = PickleSerializer(),
        use_lock: bool = True,
        allow_overwrite: bool = True,
        verbose: bool = False,
    ):
        """Constructor.

        Args:
            n: Number of samples in the list (can be larger than the number of memory slots).
            sequence: Sequence of built-in types.
            descr: Descriptor ID for shared memory access.
            slot_size: Size of a single list entry (should be big enough for even the
              biggest entry, otherwise a maximum size is estimated).
            size: Maximum cache size in MiB (if int or float); default "16.0 GiB".
            serializer: Serializer for the encoding of the dictionary data.
            use_lock: If True, applies a threading lock for multiprocessing.
            allow_overwrite: If True, cache slots can be overwritten.
            verbose: Print information to the stdout.

        Note: If the dictionary is supposed to contain the keys
          ['descr', 'size', 'sample_size', 'allow_overwrite', 'serlializer', 'verbose']
        """
        super().__init__()

        self.descr = descr
        self.serializer = serializer
        self._lock: mp.synchronize.Lock | DummyLock
        if use_lock:
            self._lock = Lock()
        else:
            self._lock = DummyLock()
        self.allow_overwrite = True
        self.verbose = verbose
        self.cache_size = (
            nbytes(f"{size}M") if isinstance(size, int | float) else nbytes(size)
        )
        if sequence:
            slot_bytes = max(
                [nbytes(slot_size)] + [serial_byte_size(v) for v in sequence]
            )
        else:
            slot_bytes = nbytes(slot_size)
        n_slots_bytes = n * slot_bytes
        cache_bytes = self.cache_size.to("B")
        states_bytes = nbytes(n * torch.uint8.itemsize)

        total_bytes = n_slots_bytes + states_bytes

        if total_bytes > cache_bytes:
            n_slots = int((cache_bytes - states_bytes) / slot_bytes)
            if self.verbose:
                print(
                    f"Requested memory ({total_bytes}) "
                    f"exceeds cache size ({self.cache_size}).\n"
                    f"Allocating cache for {n_slots} / {n} slots."
                )
        else:
            n_slots = n
            if self.verbose:
                print(
                    f"Data size ({total_bytes}) fits into "
                    f"requested cache size ({self.cache_size}).\n"
                    f"Allocating cache for {n_slots} data samples."
                )

        self._slots, self._shm_states = self.get_allocation(
            name=self.descr, n_slots=n_slots, slot_size=int(slot_bytes), size=n
        )
        for i, d in enumerate(sequence):
            self[i] = d
        self.allow_overwrite = allow_overwrite

    def get_allocation(
        self,
        name: str | None = None,
        n_slots: int | None = None,
        slot_size: int | None = None,
        size: int | None = None,
        **kwargs,
    ) -> tuple[ShareableList, torch.Tensor | None]:
        """Get a shared memory allocation.

        Args:
            name: Descriptor ID for shared memory access.
            n_slots: Number of slots the shared list should have.
            slot_size: Number of bytes each element should allocate in shared memory.
            size: Number of samples in the dataset.
            kwargs: Additional keywords for compatibility.
        """
        if name is None:
            name = self.descr
        try:
            shm_list = ShareableList(name=name)
        except FileNotFoundError:
            shm_list = ShareableList([np.random.bytes(slot_size)] * n_slots, name=name)
        if not hasattr(self, "_shm_states"):
            mp_states_arr = mp.Array(C_DTYPES[torch.uint8], size)  # type: ignore
            shm_states_arr = np.ctypeslib.as_array(mp_states_arr.get_obj())
            self._shm_states = torch.from_numpy(shm_states_arr)
            self._shm_states *= 0
            self._shm_states[n_slots:] = SlotState.OOC.value
            _shm_states = self._shm_states
        else:
            _shm_states = None
        return shm_list, _shm_states

    def clear_allocation(
        self,
    ):
        """Delete shared memory allocation."""
        if hasattr(self, "_slots"):
            try:
                self._slots.shm.close()
                self._slots.shm.unlink()
            except FileNotFoundError:
                self._slots.shm.close()

    @property
    def slots(self) -> Any:
        """Shared-memory data slots (cached sample data)."""
        return self._slots

    @property
    def states(self) -> torch.Tensor:
        """Shared-memory tensor state array.

        Note:
          - states[index] == 0 means sample at index are not yet cached.
          - states[index] == 1 means sample at index are cached.
          - states[index] == 2 means sample at index cannot be cached (due to cache limit).
        """
        return self._shm_states

    @property
    def cached_states(self) -> int:
        """Written states in cache."""
        return int(sum([s == 1 for s in self.states]))

    @property
    def cached_bytes(self) -> "nbytes":
        """Bytes written to cache."""
        n_slots = len(self.slots)
        return self.cache_size * self.cached_states / n_slots

    def get_state(self, index: int | None) -> tuple[SlotState, int | None]:
        """Get the slot state at specified index."""
        if index is None:
            return SlotState.INVALID, None
        if index < 0:
            index = len(self.states) + index
        if index < 0 or index >= len(self.states):
            raise IndexError(
                f"Index {index} out of range for dataset {list(self.states.shape)}"
            )
        return SlotState(self.states[index].item()), index

    def __getitem__(
        self, index: int | None
    ) -> int | float | bool | str | bytes | dict | None:
        """Fetch sample data from cache if stored, otherwise returns None."""
        state, index = self.get_state(index)
        if state != SlotState.SET or index is None:
            return None
        data = self._slots[index]
        if isinstance(data, bytes):
            return self.serializer.loads(data)
        return data

    def __setitem__(
        self, index: int, item: int | float | bool | str | bytes | dict | None
    ):
        """Fill the cache at specified slot."""
        state = self.get_state(index)
        if state == SlotState.OOC:
            return
        if state == SlotState.SET and not self.allow_overwrite:
            raise RuntimeError(
                f"{self} is locked and does not allow overwrites at {index=}."
            )
        if isinstance(item, dict):
            item = self.serializer.dumps(item)
        self.slots[index] = item
        self.states[index] = SlotState.SET.value

    def __contains__(self, index: int) -> bool:
        """Test 'in' cache in shared memory."""
        state = self.get_state(index)
        return state == SlotState.SET

    def __len__(self) -> int:
        """Length of the cache (may differ from dataset length depending on cache size)."""
        return len(self.slots)

    def __str__(self) -> str:
        """String of the instance."""
        n_slots = len(self.slots)
        n = len(self.states)
        return f"{self.__class__.__name__}({self.cached_states}({n_slots})/{n}@{self.cache_size.as_str()})"

    def __repr__(self) -> str:
        """Representation of the instance."""
        return self.__str__()

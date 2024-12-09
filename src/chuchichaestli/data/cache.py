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
import multiprocessing as mp
import numpy as np
import torch
from typing import Protocol


__all__ = ["SharedArray", "SharedDict", "nbytes", "get_max_ram", "get_max_shm"]


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

    def __add__(self, other: int | float):
        """Addition of nbyte instances."""
        return self.__class__(float.__add__(self, float(other)))

    def __mul__(self, other: int | float):
        """Multiplication of nbyte instances."""
        return self.__class__(float.__mul__(self, float(other)))

    def __rmul__(self, other: int | float):
        return self.__class__(float.__rmul__(self, float(other)))

    def __str__(self):
        """String of instance."""
        return self.as_bstr()

    def __repr__(self):
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

    def to(self, unit: str) -> float:
        """Convert to unit."""
        if unit in self.units:
            return self.__class__(self / self.units[unit])
        else:
            raise ValueError(f"Unknown unit, choose from {list(self.units.keys())}.")


class DictSerializer(Protocol):
    def dumps(self, dct: dict) -> bytes:
        ...

    def loads(self, data: bytes) -> dict:
        ...


class PickleSerializer:
    def dumps(self, dct: dict) -> bytes:
        return pickle.dumps(dct, pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> dict:
        return pickle.loads(data)


def get_max_ram() -> nbytes:
    """Get the maximal size of total RAM."""
    return nbytes(psutil.virtual_memory().total)


def get_max_shm() -> nbytes:
    """Get the maximal size of available shared memory."""
    return nbytes(psutil.Process().memory_info().rss)


def estimate_dict_size(
    dct: dict,
    serializer: type[DictSerializer] = PickleSerializer,
) -> nbytes:
    """Estimate the size of a dictionary with mixed values."""
    enc_bstr = serializer().dumps(dct)
    return nbytes(len(enc_bstr))


class SlotState(Enum):
    EMPTY = 0
    SET = 1
    OOC = 2  # not in cache but valid dataset index


class SharedArray:
    """A shared memory cache array for use as tensor data cache in PyTorch datasets.

    Wrapper class for multiprocessing.Array, a ctypes array on shared memory.
    If the size of the dataset exceeds the size of the set cache limit,
    only the first N samples will be cached. For shuffled datasets,
    this is called 'stochastic caching'.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        size: int | float | str = "4.0G",
        dtype: torch.dtype = torch.float32,
        allow_overwrite: bool = False,
        verbose: bool = False,
    ):
        """Constructor.

        Args:
          shape: Dataset N-d shape, e.g. (n_samples, channels, width, height, depth).
          size: Maximum cache size in GiB (if int or float); default "4.0 GiB".
          dtype: PyTorch data type (default torch.float32). Must be type with corresponding ctype,
            i.e. bool, uint8, int8/16/32/64, or float32/64.
          allow_overwrite: If True, cache slots can be overwritten.
          verbose: Print information to the stdout.
        """
        cache_size = nbytes(f"{size}G") if isinstance(size, int | float) else nbytes(size)
        self.allow_overwrite = allow_overwrite
        self.verbose = verbose

        if dtype not in C_DTYPES:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be one of {C_DTYPES.keys()}"
            )
        slot_size = int(np.prod(shape[1:]))
        slot_bytes = nbytes(slot_size * nbytes(dtype.itemsize))
        dataset_bytes = nbytes(shape[0] * slot_bytes)
        cache_bytes = cache_size.to("B")
        states_bytes = nbytes(shape[0] * torch.uint8.itemsize)

        total_bytes = dataset_bytes + states_bytes

        if total_bytes > cache_bytes:
            n_slots = int((cache_bytes - states_bytes) / slot_bytes)
            if self.verbose:
                print(
                    f"Requested memory ({total_bytes}) "
                    f"exceeds cache size ({cache_size}).\n"
                    f"Allocating cache for {n_slots} / {shape[0]} data samples."
                )
        else:
            n_slots = shape[0]
            if self.verbose:
                print(
                    f"Dataset size ({total_bytes}) fits into "
                    f"requested cache size ({cache_size}).\n"
                    f"Allocating cache for {n_slots} data samples."
                )

        mp_arr = mp.Array(C_DTYPES[dtype], n_slots * slot_size)  # type: ignore
        shm_arr = np.ctypeslib.as_array(mp_arr.get_obj())
        shm_arr = shm_arr.reshape((n_slots, *shape[1:]))
        self._t_shm = torch.from_numpy(shm_arr)
        self._t_shm *= 0

        mp_states_arr = mp.Array(C_DTYPES[torch.uint8], shape[0])  # type: ignore
        shm_states_arr = np.ctypeslib.as_array(mp_states_arr.get_obj())
        self._t_states = torch.from_numpy(shm_states_arr)
        self._t_states *= 0
        self._t_states[n_slots:] = SlotState.OOC.value

    @property
    def array(self) -> torch.Tensor:
        """Shared-memory tensor data array (cached samples)."""
        return self._t_shm

    @property
    def states(self) -> torch.Tensor:
        """Shared-memory tensor state array.

        Note:
          - states[index] == 0 means sample at index are not yet cached.
          - states[index] == 1 means sample at index are cached.
          - states[index] == 2 means sample at index cannot be cached (due to cache limit).
        """
        return self._t_states

    def get_state(self, index: int):
        """Get the slot state at specified index."""
        if index < 0:
            index = len(self.states) + index
        if index < 0 or index >= len(self.states):
            raise IndexError(f"Index {index} out of range for dataset {list(self.states.shape)}")
        return SlotState(self.states[index].item())

    def clear(self, index: int | None = None):
        """Clear the cache (optionally only at a specified index)."""
        if index is None:
            self._t_shm *= 0
            self._t_states *= 0
            self._t_states[len(self):] = SlotState.OOC.value
        else:
            _ = self.get_state(index)
            self[index] = torch.zeros(self._t_shm.shape[1:])
            self._t_states[index] = torch.zeros(self._t_states.shape[1:])
            if len(self) <= index:
                self._t_states[index] = SlotState.OOC.value

    def __getitem__(self, index: int) -> torch.Tensor | None:
        """Fetch sample tensor from cache if stored, otherwise returns None."""
        state = self.get_state(index)
        if state == SlotState.EMPTY or state == SlotState.OOC:
            return None
        return self.array[index]

    def __setitem__(self, index: int, item: torch.Tensor):
        """Fill the cache at specified location."""
        state = self.get_state(index)
        if state == SlotState.OOC:
            return
        if state == SlotState.SET and self.allow_overwrite:
            raise RuntimeError(
                f"SharedCache is locked and does not allow overwrites at {index=}."
            )
        self.array[index] = item
        self.states[index] = SlotState.SET.value

    def __len__(self) -> int:
        """Length of the cache (may differ from dataset length)."""
        return len(self.array)


class SharedDict:
    """A shared memory cache dictionary for use as metadata cache in PyTorch datasets."""

    def __init__(
        self,
        descr: str = "shm_dict",
        size: int | float | str = "16M",
        allow_overwrite: bool = False,
        serializer: DictSerializer = PickleSerializer(),
        verbose: bool = False,
        **kwargs
    ):
        """Constructor.

        Args:
          descr: Descriptor ID for shared memory access.
          size: Maximum cache size in MiB (if int or float); default "16.0 MiB".
          allow_overwrite: If True, cache slots can be overwritten.
          verbose: Print information to the stdout.

        Note: If the dictionary is supposed to contain the keys
          ['descr', 'size', 'sample_size', 'allow_overwrite', 'serlializer', 'verbose']
        """
        cache_size = nbytes(f"{size}M") if isinstance(size, int | float) else nbytes(size)
        self.allow_overwrite = allow_overwrite
        self.verbose = verbose
        
    

class SharedDictList:
    """A shared memory cache dictionary list for use as metadata cache in PyTorch datasets.

    If the size of the dataset's metadata exceeds the size of the set cache limit,
    only the first N samples will be cached. For shuffled datasets, this is called
    'stochastic caching'.

    Note: If PyTorch tensors are to be used, rather opt for the SharedArray class.
    """

    def __init__(
        self,
        descr: str = "meta_shm",
        size: int | float | str = "16M",
        sample_size: int | float | None = None,
        n_blocks: int = 1,
        allow_overwrite: bool = False,
        serializer: DictSerializer = PickleSerializer(),
        verbose: bool = False,
        **kwargs
    ):
        """Constructor.

        Args:
          descr: Descriptor ID for shared memory access.
          size: Maximum cache size in MiB (if int or float); default "16.0 MiB".
          sample_size: Maximum byte size of a single dict entry
          sample_dict: Alternative argument to `sample_size`, estimates
            maximum byte size of a dict entry.
          allow_overwrite: If True, cache slots can be overwritten.
          verbose: Print information to the stdout.

        Note: If the dictionary is supposed to contain the keys
          ['descr', 'size', 'sample_size', 'allow_overwrite', 'serlializer', 'verbose']
        """
        cache_size = nbytes(f"{size}M") if isinstance(size, int | float) else nbytes(size)
        block_size = cache_size
        self.allow_overwrite = allow_overwrite
        self.verbose = verbose
        print(cache_size)
        # TODO


if __name__ == "__main__":
    import sys
    import numpy as np
    metadata = {
        'N_particle_flag': 0,
        'box': 'tng50-1',
        'class': 'dm',
        'extent': np.array([
            -243.30606099,  243.30606099, -243.30606099,  243.30606099
        ]),
        'gid': 0,
        'has_bh': 1,
        'name': 'dm_tng50-1.50.gid.0000000',
        'num_particles': 57709743,
        'rng_seed': 42,
        'rotxy': np.array([0, 0]),
        'simulation': 'IllustrisTNG',
        'snapshot': 50,
        'units': 'solMass / kpc2',
        'units_extent': 'kpc'
    }

    enc_size = estimate_dict_size(metadata)
    print(enc_size.as_str())
    print((enc_size * 12000).as_str())
    print(len(b'1'))

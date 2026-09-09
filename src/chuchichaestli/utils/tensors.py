# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small tensor helpers shared across chuchichaestli."""

from typing import Any

import numpy as np
import torch


__all__ = [
    "as_array",
    "as_inexact",
    "npy_to_torch_dtype",
    "torch_to_npy_dtype",
    "view_along_axis",
]


NPY_TO_TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def npy_to_torch_dtype(dtype: str | np.dtype | type) -> torch.dtype | None:
    """Converts numpy dtype to torch dtype robustly.

    Args:
        dtype: A numpy dtype, a numpy type, or the name of either.
    """
    try:
        name = np.dtype(dtype).name  # e.g. "uint8", "bool"
    except Exception:
        name = str(dtype)
    return NPY_TO_TORCH_DTYPES.get(name)


def torch_to_npy_dtype(dtype: torch.dtype) -> np.dtype:
    """Return the numpy dtype matching a torch dtype.

    Args:
        dtype: Torch dtype to translate.
    """
    return np.dtype(torch.empty((), dtype=dtype).numpy().dtype)


def as_array(x: Any) -> np.ndarray:
    """Return the data as a numpy array, detached and on the host.

    A tensor is handed over as a view wherever numpy can share its memory,
    which a tensor on the host and outside a graph always can.

    Args:
        x: Tensor, array, or anything numpy can read as one.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_inexact(x: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Promote an integer or boolean tensor to a floating point one.

    Args:
        x: Input tensor.
        dtype: Floating point type integer and boolean input is promoted to.
    """
    return x if x.is_floating_point() or x.is_complex() else x.to(dtype)


def view_along_axis(values: torch.Tensor, ndim: int, axis: int) -> torch.Tensor:
    """Reshape a vector so it broadcasts along `axis` of an `ndim`-rank tensor.

    Args:
        values: One value per entry along `axis`.
        ndim: Rank of the tensor the vector is broadcast against.
        axis: Axis the vector runs along.
    """
    shape = [1] * ndim
    shape[axis] = values.numel()
    return values.reshape(shape)

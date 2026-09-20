# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small tensor helpers shared across chuchichaestli."""

import numpy as np
import numpy.typing as npt
import torch


__all__ = [
    "as_array",
    "as_batched_slices",
    "as_inexact",
    "as_tri_channel",
    "sanitize_ndim",
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


def as_array(x: torch.Tensor | npt.ArrayLike) -> np.ndarray:
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


def sanitize_ndim(x: torch.Tensor, check_2D: bool = True, check_3D: bool = False):
    """Standardize image dimensionality to (B, C, W, H)."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    if check_2D and check_3D and (x.ndim != 4 and x.ndim != 5):
        raise ValueError(
            f"Require input of shape {'(C, W, H) or (B, C, W, H)' if check_2D else ''}"
            f"{' or (B, C, W, H, D)' if check_3D else ''}."
        )
    elif check_3D and not check_2D and x.ndim != 5:
        raise ValueError("Require input of shape (B, C, W, H, D).")
    elif check_2D and not check_3D and x.ndim != 4:
        raise ValueError("Require input of shape (C, W, H) or (B, C, W, H).")
    return x


def as_tri_channel(x: torch.Tensor):
    """Morph input to resemble a three-channel image."""
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] < 3:
        x = x[:, 0:1, :, :].repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        raise ValueError(f"Input has more than three channels ({x.shape[1]})!")
    return x


def as_batched_slices(x: torch.Tensor, sample: int = 0) -> torch.Tensor:
    """Convert batches of volumetric 5D tensors into 4D slice-wise image tensors.

    Args:
        x: Volumetric 5D input tensor.
        sample: If `> 0`, the volume depth is sampled `sample` times from the centre.
    """
    if x.ndim == 5:
        B, C, W, H, D = x.shape
        if sample > 0:
            sample = min(sample, D)
            center = D // 2
            window = sample // 2
            start = center - window
            end = start + sample
            if sample % 2 == 0:
                start = center - window
                end = center + window
            x = x[..., start:end]
            D = sample
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * D, C, W, H)
    return x

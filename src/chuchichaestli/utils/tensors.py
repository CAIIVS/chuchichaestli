# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small tensor helpers shared across chuchichaestli."""

import torch


__all__ = ["as_inexact", "view_along_axis"]


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

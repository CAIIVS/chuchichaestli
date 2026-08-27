# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Complex/real view transforms."""

from typing import Any
import torch
from torchvision.transforms.v2 import Transform

__all__ = ["ComplexExpand", "ComplexCollapse"]


class ComplexExpand(Transform):
    """Expand a complex tensor into a real one with an axis of 2.

    The real/imaginary axis is appended by `torch.view_as_real` and then moved
    to `dim`, so it is trailing only for the default `dim=-1`.

    Use this if complex data should work with `SharedArray`.
    Both directions are zero-copy views wherever the strides allow it; only
    `revert` on a non-contiguous input falls back to a copy.

    Note that the real dtype follows the complex one:
        - `complex128` expands to `float64`
        - `complex64` expands to `float32`
        - `complex32` expands to `float16` (incompatible with `SharedArray`)
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, dim: int = -1, strict: bool = True):
        """Constructor.

        Args:
            dim: Position of the real/imaginary axis in the expanded tensor.
                The default `-1` is the natural layout of `torch.view_as_real`;
                any other position adds a `movedim` (still a view). Use `0` or
                `1` to feed real and imaginary parts to a network as channels.
            strict: If `True`, a non-complex input to `transform` (or an input
                to `revert` whose `dim` axis is not of size 2) raises. Set
                `False` to pass such tensors through unchanged, which is what
                you want when the transform is applied across a nested sample
                holding both complex and real entries -- at the cost of the
                round trip no longer being checked.
        """
        super().__init__()
        self.dim = dim
        self.strict = strict

    def _expand(self, x: Any) -> Any:
        """Expand complex to real, placing the real/imaginary axis at `dim`."""
        if not torch.is_complex(x):
            if self.strict:
                raise TypeError(
                    f"expected a complex tensor, got {x.dtype}; pass "
                    "`strict=False` to let real tensors through unchanged"
                )
            return x
        rank = x.ndim + 1  # view_as_real appends the real/imaginary axis
        if not -rank <= self.dim < rank:
            raise ValueError(
                f"dim {self.dim} is out of range for the expanded tensor, "
                f"which has {rank} axes"
            )
        return torch.view_as_real(x).movedim(-1, self.dim)

    def _collapse(self, x: Any) -> Any:
        """Collapse the size-2 axis at `dim` back into a complex tensor."""
        if torch.is_complex(x):
            if self.strict:
                raise TypeError(f"expected a real tensor, got {x.dtype}")
            return x
        in_range = -x.ndim <= self.dim < x.ndim
        if not in_range or x.shape[self.dim] != 2:
            if self.strict:
                raise ValueError(
                    f"expected axis {self.dim} to be of size 2, got shape "
                    f"{tuple(x.shape)}; pass `strict=False` to let such "
                    "tensors through unchanged"
                )
            return x
        x = x.movedim(self.dim, -1)
        try:
            return torch.view_as_complex(x)
        except RuntimeError:
            # view_as_complex needs a unit stride on the last axis
            return torch.view_as_complex(x.contiguous())

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Expand complex to real (ignores `params`)."""
        return self._expand(x)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Collapse real back to complex (ignores `params`)."""
        return self._collapse(x)


class ComplexCollapse(ComplexExpand):
    """Inverse of `ComplexExpand`: collapse to complex, expand on revert."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Collapse real to complex."""
        return self._collapse(x)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Expand complex to real."""
        return self._expand(x)

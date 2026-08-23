# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Composition of v2 transforms with a single shared per-batch param draw."""

from typing import Any
from collections.abc import Sequence
from torchvision.transforms.v2 import Transform

__all__ = ["SequentialTransform"]


class SequentialTransform(Transform):
    """Chain v2 transforms, sampling each child's params once per batch.

    Unlike `torchvision`'s `Compose` (which re-samples per call and exposes no
    `make_params`/`transform`), this drives the child `make_params`/`transform`
    hooks directly, so it plugs into `SequenceCollate` and applies the *same*
    draw to every paired field/step. It is invertible when all children are.
    """

    _transformed_types = (Transform,)  # placeholder; children gate their own types

    def __init__(self, *transforms: Transform | Sequence[Transform]):
        """Constructor.

        Args:
            transforms: The child transforms to apply in order, given either as
                positional arguments or a single sequence.
        """
        super().__init__()
        if len(transforms) == 1 and isinstance(transforms[0], list | tuple):
            transforms = tuple(transforms[0])
        self.transforms = list(transforms)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample each child's params once, threading the running leaves through.

        Later children see the (intermediate) shape produced by earlier ones, so
        e.g. a crop before a flip is accounted for.
        """
        params, cur = [], list(flat_inputs)
        for t in self.transforms:
            p = t.make_params(cur)
            params.append(p)
            cur = [t.transform(x, p) for x in cur]
        return {"params": params}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply every child in order with its shared params."""
        for t, p in zip(self.transforms, params["params"]):
            inpt = t.transform(inpt, p)
        return inpt

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Invert every child in reverse order (all children must be invertible)."""
        for t in reversed(self.transforms):
            x = t.revert(x)
        return x

    def get_inverse(self) -> "SequentialTransform":
        """Return the reversed chain of child inverses."""
        return SequentialTransform(*[t.get_inverse() for t in reversed(self.transforms)])

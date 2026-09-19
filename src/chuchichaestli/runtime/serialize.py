# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning what a run holds into a storable form, and back again."""

from __future__ import annotations
import warnings
from collections.abc import Mapping
from typing import Any
import torch
from chuchichaestli.models.spec import ModelSpec
from chuchichaestli.runtime.traits import Stage


__all__ = [
    "C3liCheckpointError",
    "unpack_tree",
    "merge_tree",
    "stage_signature",
    "writable_spec",
]


_TENSOR_TAG = "__tensor__"
_PAIRS_TAG = "__pairs__"


class C3liCheckpointError(ValueError):
    """Raised when a checkpoint cannot be written, found or trusted."""


def _child(path: str, key: Any) -> str:
    """Return the path of a child within the tree.

    Args:
        path: Path of the parent, empty at the root.
        key: Mapping key or sequence index of the child.
    """
    return f"{path}/{key}" if path else str(key)


def _strip_tensors(value: Any, path: str, tensors: dict[str, torch.Tensor]) -> Any:
    """Replace tensors with references and record them under their path.

    Args:
        value: Part of the tree to encode.
        path: Slash-joined position of `value` within the tree.
        tensors: Collects every tensor found, keyed by its path.
    """
    if isinstance(value, torch.Tensor):
        tensors[path] = value
        return {_TENSOR_TAG: path}
    if isinstance(value, Mapping):
        items = [
            (key, _strip_tensors(item, _child(path, key), tensors))
            for key, item in value.items()
        ]
        if all(isinstance(key, str) for key in value):
            return dict(items)
        return {_PAIRS_TAG: [[key, item] for key, item in items]}
    if isinstance(value, (list, tuple)):
        return [
            _strip_tensors(item, _child(path, index), tensors)
            for index, item in enumerate(value)
        ]
    return value


def _restore_tensors(value: Any, tensors: Mapping[str, torch.Tensor]) -> Any:
    """Put the referenced tensors back into a skeleton.

    Args:
        value: Part of the skeleton to decode.
        tensors: Tensors keyed by the path their reference names.

    Raises:
        C3liCheckpointError: If a reference names a tensor the file lacks.
    """
    if isinstance(value, Mapping):
        if len(value) == 1 and _TENSOR_TAG in value:
            key = value[_TENSOR_TAG]
            if key not in tensors:
                raise C3liCheckpointError(
                    f"The manifest refers to a tensor {key!r} that the "
                    "checkpoint's tensor file does not hold."
                )
            return tensors[key]
        if len(value) == 1 and _PAIRS_TAG in value:
            return {
                key: _restore_tensors(item, tensors) for key, item in value[_PAIRS_TAG]
            }
        return {key: _restore_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_tensors(item, tensors) for item in value]
    return value


def unpack_tree(tree: Any) -> tuple[dict[str, torch.Tensor], Any]:
    """Split a tree into flat tensors and a JSON-serializable skeleton.

    Mappings with non-string keys are kept as pairs, so an optimizer's integer
    parameter indices survive JSON.

    Args:
        tree: The tree to split.
    """
    tensors: dict[str, torch.Tensor] = {}
    return tensors, _strip_tensors(tree, "", tensors)


def merge_tree(tensors: Mapping[str, torch.Tensor], skeleton: Any) -> Any:
    """Rebuild the tree that `unpack_tree` split.

    Args:
        tensors: Tensors as read back from the checkpoint's tensor file.
        skeleton: The tree shape, as returned by `unpack_tree`.
    """
    return _restore_tensors(skeleton, tensors)


def stage_signature(stage: Stage) -> list[list[str]]:
    """Return the program's stages as `[path, class name]` pairs.

    Args:
        stage: Root of the stage tree, usually the program.
    """
    walk = getattr(stage, "walk", None)
    if callable(walk):
        return [[path, name] for path, name in walk()]
    return [[stage.name, type(stage).__name__]]


def writable_spec(value: Any) -> ModelSpec | None:
    """Return a component's model spec if it records one that can be written.

    Args:
        value: Component to ask.
    """
    spec = getattr(value, "spec", None)
    if not isinstance(spec, ModelSpec):
        return None
    try:
        spec.to_dict()
    except TypeError as exc:
        warnings.warn(
            f"Not recording a model spec: {exc}",
            stacklevel=2,
        )
        return None
    return spec

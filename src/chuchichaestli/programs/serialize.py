# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning what a run holds into a storable form, and back again."""

from __future__ import annotations
import dataclasses
import functools
import warnings
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from pathlib import Path
from typing import Any
import torch
from torch import nn
from chuchichaestli.models.spec import ModelSpec
from chuchichaestli.programs.context import subpath, split_path
from chuchichaestli.programs.traits import Stage, Stateful


__all__ = [
    "C3liCheckpointError",
    "unpack_tree",
    "merge_tree",
    "stage_signature",
    "stages_by_path",
    "component_states",
    "hook_keys",
    "hook_states",
    "writable_spec",
    "as_loggable",
    "stage_settings",
]


_TENSOR_TAG = "__tensor__"
# longer sequences are summarised by their length rather than listed
_DESCRIBED_ITEMS = 32
# constructor arguments that identify a stage or wire it, rather than set it up
_WIRING = frozenset({"name", "requires", "provides"})
_PAIRS_TAG = "__pairs__"


class C3liCheckpointError(ValueError):
    """Raised when a checkpoint cannot be written, found or trusted."""


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
            (key, _strip_tensors(item, subpath(path, key), tensors))
            for key, item in value.items()
        ]
        if all(isinstance(key, str) for key in value):
            return dict(items)
        return {_PAIRS_TAG: [[key, item] for key, item in items]}
    if isinstance(value, (list, tuple)):
        return [
            _strip_tensors(item, subpath(path, index), tensors)
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


def stages_by_path(stage: Stage) -> dict[str, Stage]:
    """Return each distinct stage of a tree, keyed by where it first appears.

    A repeated stage holds one object in every slot, so it is keyed once.

    Args:
        stage: Root of the stage tree, usually the program.
    """
    walk = getattr(stage, "walk", None)
    if not callable(walk):
        return {stage.name: stage}
    found: dict[str, Stage] = {}
    seen: set[int] = set()
    for path, _ in walk():
        held = _stage_at(stage, path)
        if held is None or id(held) in seen:
            continue
        seen.add(id(held))
        found[path] = held
    return found


def _stage_at(root: Stage, path: str) -> Stage | None:
    """Return the stage a walked path names.

    Args:
        root: Root of the stage tree.
        path: Path as `walk` spells it, e.g. `"program/0:fit"`.
    """
    held = root
    for index, _ in split_path(path)[1:]:
        children = getattr(held, "stages", ())
        if index >= len(children):
            return None
        held = children[index]
    return held


def component_states(stage: Stage) -> dict[str, Any]:
    """Return the component states of each stage of a tree, keyed by its path.

    Args:
        stage: Root of the stage tree, usually the program.
    """
    states: dict[str, Any] = {}
    for path, held in stages_by_path(stage).items():
        capture = getattr(held, "component_states", None)
        if callable(capture):
            states[path] = capture()
    return states


def hook_keys(hooks: Sequence[Any]) -> list[str]:
    """Return the name each hook's state is checkpointed under.

    Args:
        hooks: The runner's hooks, in order.
    """
    keys: list[str] = []
    seen: dict[str, int] = {}
    for hook in hooks:
        name = getattr(hook, "state_key", None) or type(hook).__name__
        keys.append(f"{name}#{seen.get(name, 0)}")
        seen[name] = seen.get(name, 0) + 1
    return keys


def hook_states(hooks: Sequence[Any]) -> dict[str, Any]:
    """Return the state of each hook that keeps one, keyed by `hook_keys`.

    Args:
        hooks: The runner's hooks, in order.
    """
    return {
        key: hook.state_dict()
        for key, hook in zip(hook_keys(hooks), hooks)
        if isinstance(hook, Stateful)
    }


def writable_spec(value: Any) -> ModelSpec | None:
    """Return a component's model spec if it records one that can be written.

    Args:
        value: Component to ask.
    """
    spec = getattr(value, "spec", None)
    if not isinstance(spec, ModelSpec):
        return None
    try:
        spec.to_json()
    except TypeError as exc:
        warnings.warn(
            f"Not recording a model spec: {exc}",
            stacklevel=2,
        )
        return None
    return spec


def as_loggable(value: Any) -> Any:
    """Return a readable, JSON-safe account of a value, lossy where it must be.

    Args:
        value: What to log; anything unrecognised is named by its class.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype).removeprefix("torch.")
        return f"tensor(shape={list(value.shape)}, dtype={dtype})"
    recorded = getattr(value, "init_args", None)
    if isinstance(recorded, Mapping):
        return {"cls": type(value).__name__, "kwargs": as_loggable(recorded)}
    if isinstance(value, nn.Module):
        return type(value).__name__
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: as_loggable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): as_loggable(v) for k, v in value.items()}
    if isinstance(value, Set):
        return sorted((as_loggable(v) for v in value), key=str)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        if len(value) > _DESCRIBED_ITEMS:
            return f"{type(value).__name__} of {len(value)} items"
        return [as_loggable(v) for v in value]
    if isinstance(value, functools.partial):
        return as_loggable(value.func)
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', '')}.{value.__qualname__}".lstrip(".")
    return type(value).__name__


def stage_settings(stage: Stage) -> dict[str, Any]:
    """Return the arguments a stage was constructed with, readable and JSON-safe.

    Args:
        stage: The stage to describe; one recording nothing gives `{}`.
    """
    recorded = getattr(stage, "init_args", None) or {}
    return {
        key: as_loggable(value)
        for key, value in recorded.items()
        if key not in _WIRING and not _holds_stages(value)
    }


def _holds_stages(value: Any) -> bool:
    """Whether a value is a stage or a sequence of them, which report their own.

    Args:
        value: Constructor argument to check.
    """
    if isinstance(value, Stage):
        return True
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and bool(value)
        and all(isinstance(v, Stage) for v in value)
    )

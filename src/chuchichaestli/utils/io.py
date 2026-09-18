# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filesystem helpers for reading and writing states."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from chuchichaestli.models.spec import ModelSpec
from chuchichaestli.utils.registry import require


_CANNOT = "Cannot {action} '{{name}}'; choose from {{options}}."

METADATA_KEY = "__metadata__"
MODELSPEC_KEY = "spec"

__all__ = [
    "METADATA_KEY",
    "MODELSPEC_KEY",
    "staged",
    "READERS",
    "WRITERS",
    "reader_for",
    "writer_for",
    "read_state",
    "read_spec",
    "write_state",
    "load_model",
]


@contextmanager
def staged(*targets: Path) -> Iterator[list[Path]]:
    """Yield a scratch path per target; all are moved into place, or none are.

    Args:
        targets: Files the write is to land on.
    """
    scratches = [t.with_stem(f"{t.stem}.part") for t in targets]
    try:
        yield scratches
    except BaseException:
        for scratch in scratches:
            scratch.unlink(missing_ok=True)
        raise
    for scratch, target in zip(scratches, targets):
        scratch.replace(target)


def _read_safetensors(path: Path, spec_only: bool = False) -> Any:
    """Read a safetensors file as `(state, spec)`, or the spec on its own.

    Reading the spec alone touches the metadata header rather than the
    tensors, so a large checkpoint can be inspected cheaply.

    Args:
        path: File to read.
        spec_only: Return the stored spec instead of the pair.
    """
    with safe_open(str(path), framework="pt") as handle:
        stored = (handle.metadata() or {}).get(MODELSPEC_KEY)
    if spec_only:
        return stored
    return load_file(str(path)), stored


def _read_torch(path: Path, spec_only: bool = False) -> Any:
    """Read a torch archive as `(state, spec)`, or the spec on its own.

    Refuses to unpickle anything but tensors and plain strings.

    Args:
        path: File to read.
        spec_only: Return the stored spec instead of the pair.
    """
    stored = torch.load(path, map_location="cpu", weights_only=True)
    metadata = stored.get(METADATA_KEY, {})
    if spec_only:
        return metadata.get(MODELSPEC_KEY)
    state = {k: v for k, v in stored.items() if k != METADATA_KEY}
    return state, metadata.get(MODELSPEC_KEY)


def _write_safetensors(
    path: Path, state: dict[str, torch.Tensor], spec: str | None = None
) -> None:
    """Write a safetensors file, keeping any spec in its metadata header.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: Serialized spec to store beside them, if any.
    """
    save_file(state, str(path), metadata={MODELSPEC_KEY: spec} if spec else None)


def _write_torch(
    path: Path, state: dict[str, torch.Tensor], spec: str | None = None
) -> None:
    """Write a torch archive, keeping any spec under a reserved key.

    An archive has no metadata header, so one is stored beside the tensors
    under a reserved key, holding the spec in the same field safetensors uses.
    `weights_only` accepts a mapping of plain strings, and the reader strips
    the reserved key out again.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: Serialized spec to store beside them, if any.
    """
    torch.save({**state, METADATA_KEY: {MODELSPEC_KEY: spec}} if spec else state, path)


READERS: dict[str, Callable[..., Any]] = {
    ".safetensors": _read_safetensors,
    ".pt": _read_torch,
    ".pth": _read_torch,
}

WRITERS: dict[str, Callable[..., None]] = {
    ".safetensors": _write_safetensors,
    ".pt": _write_torch,
    ".pth": _write_torch,
}


def reader_for(path: Path) -> Callable[..., Any]:
    """Return the reader for a path's suffix.

    Exposed alongside `read_state` so a caller can reject a path before doing
    work that only one process should do.

    Args:
        path: File whose suffix selects the reader.
    """
    return require(path.suffix, READERS, message=_CANNOT.format(action="read"))


def writer_for(path: Path) -> Callable[..., None]:
    """Return the writer for a path's suffix.

    Args:
        path: File whose suffix selects the writer.
    """
    return require(path.suffix, WRITERS, message=_CANNOT.format(action="write"))


def read_state(
    path: Path, spec: bool = False
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ModelSpec | None]:
    """Read a state dict, picking the reader from the suffix.

    Args:
        path: File to read.
        spec: Return `(state, spec)` rather than the state alone. The spec is
            `None` when the file carries none.
    """
    state, stored = reader_for(path)(path)
    if not spec:
        return state
    return state, ModelSpec.from_json(stored) if stored else None


def read_spec(path: Path) -> ModelSpec | None:
    """Return the spec a file carries, or `None` if it carries none.

    Args:
        path: File to read.
    """
    stored = reader_for(path)(path, spec_only=True)
    return ModelSpec.from_json(stored) if stored else None


def write_state(
    path: Path, state: dict[str, torch.Tensor], spec: ModelSpec | None = None
) -> None:
    """Write a state dict, picking the writer from the suffix.

    Tensors are detached onto the CPU and made contiguous on the way out.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: How to rebuild the model, stored beside the tensors.

    Raises:
        ValueError: If the suffix names no known format.
    """
    storable = {
        key: value.detach().cpu().contiguous()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in state.items()
    }
    writer_for(path)(path, storable, spec.to_json() if spec is not None else None)


def load_model(path: Path, strict: bool = True, **overrides: Any) -> Any:
    """Rebuild a model from a file and load its weights.

    Args:
        path: File to read.
        strict: Whether every key must match, as for `load_state_dict`.
        overrides: Constructor arguments to replace before building. Changing
            one that affects a tensor's shape makes the weights unloadable,
            which `strict` does not cover.

    Raises:
        ValueError: If the file carries no spec, since the architecture cannot
            be inferred from weights alone.
    """
    state, spec = read_state(path, spec=True)
    if spec is None:
        raise ValueError(
            f"No model spec in {str(path)!r}; weights alone do not say which "
            "architecture built them. Save it with write_state(..., spec=...)."
        )
    model = spec.build(**overrides)
    model.load_state_dict(state, strict=strict)
    return model

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

from chuchichaestli.utils.registry import require


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
    "write_state",
]


@contextmanager
def staged(*targets: Path) -> Iterator[list[Path]]:
    """Yield a scratch path per target, each moved into place if the block succeeds.

    Args:
        targets: Files the write is to land on.
    """
    scratches = [t.with_stem(f"{t.stem}.part") for t in targets]
    try:
        yield scratches
        for scratch, target in zip(scratches, targets):
            scratch.replace(target)
    finally:
        for scratch in scratches:
            scratch.unlink(missing_ok=True)


def _read_safetensors(
    path: Path, spec_only: bool = False
) -> str | None | tuple[dict[str, torch.Tensor], str | None]:
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


def _read_torch(
    path: Path, spec_only: bool = False
) -> str | None | tuple[dict[str, torch.Tensor], str | None]:
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
    return require(
        path.suffix,
        READERS,
        message=lambda options: f"Cannot read '{path.suffix}'; choose from {options}.",
    )


def writer_for(path: Path) -> Callable[..., None]:
    """Return the writer for a path's suffix.

    Args:
        path: File whose suffix selects the writer.
    """
    return require(
        path.suffix,
        WRITERS,
        message=lambda options: f"Cannot write '{path.suffix}'; choose from {options}.",
    )


def read_state(path: Path) -> dict[str, torch.Tensor]:
    """Read a state dict, picking the reader from the suffix.

    Args:
        path: File to read.
    """
    state, _ = reader_for(path)(path)
    return state


def write_state(
    path: Path, state: dict[str, torch.Tensor], spec: str | None = None
) -> None:
    """Write a state dict, picking the writer from the suffix.

    Tensors are detached onto the CPU and made contiguous on the way out.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: Serialized spec to store beside them, if any.

    Raises:
        ValueError: If the suffix names no known format.
    """
    storable = {
        key: value.detach().cpu().contiguous()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in state.items()
    }
    writer_for(path)(path, storable, spec)

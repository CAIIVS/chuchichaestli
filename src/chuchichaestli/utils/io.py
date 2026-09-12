# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filesystem helpers for reading and writing states."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from chuchichaestli.models.spec import ModelSpec


SPEC_KEY = "__metadata__"

__all__ = [
    "SPEC_KEY",
    "staged",
    "READERS",
    "WRITERS",
    "reader_for",
    "writer_for",
    "read_state",
    "write_state",
    "SPEC_READERS",
    "SPEC_WRITERS",
    "read_spec",
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


def _read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Read a safetensors file.

    Args:
        path: File to read.
    """
    return load_file(str(path))


def _read_torch(path: Path) -> dict[str, torch.Tensor]:
    """Read a torch archive, refusing to unpickle anything but tensors.

    Args:
        path: File to read.
    """
    stored = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in stored.items() if k != SPEC_KEY}


def _write_safetensors(path: Path, state: dict[str, torch.Tensor]) -> None:
    """Write a safetensors file.

    Args:
        path: File to write.
        state: Tensors to store.
    """
    save_file(state, str(path))


def _write_torch(path: Path, state: dict[str, torch.Tensor]) -> None:
    """Write a torch archive.

    Args:
        path: File to write.
        state: Tensors to store.
    """
    torch.save(state, path)


def _read_torch_spec(path: Path) -> str | None:
    """Return the spec stored beside the tensors, if there is one.

    Args:
        path: File to read.
    """
    stored = torch.load(path, map_location="cpu", weights_only=True)
    return stored.get(SPEC_KEY)


def _read_safetensors_spec(path: Path) -> str | None:
    """Return the spec from the file's header, if there is one.

    Args:
        path: File to read.
    """
    with safe_open(str(path), framework="pt") as handle:
        return (handle.metadata() or {}).get("spec")


def _write_torch_spec(path: Path, state: dict[str, torch.Tensor], spec: str) -> None:
    """Write tensors with the spec alongside them under a reserved key.

    A torch archive has no metadata header, but it stores any picklable value,
    and `weights_only` accepts a plain string.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: Serialized spec to store beside them.
    """
    torch.save({**state, SPEC_KEY: spec}, path)


def _write_safetensors_spec(
    path: Path, state: dict[str, torch.Tensor], spec: str
) -> None:
    """Write tensors with the spec in the file's metadata header.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: Serialized spec to store beside them.
    """
    save_file(state, str(path), metadata={"spec": spec})


READERS: dict[str, Callable[[Path], dict[str, torch.Tensor]]] = {
    ".safetensors": _read_safetensors,
    ".pt": _read_torch,
    ".pth": _read_torch,
}

WRITERS: dict[str, Callable[[Path, dict[str, torch.Tensor]], None]] = {
    ".safetensors": _write_safetensors,
    ".pt": _write_torch,
    ".pth": _write_torch,
}

SPEC_READERS: dict[str, Callable[[Path], str | None]] = {
    ".safetensors": _read_safetensors_spec,
    ".pt": _read_torch_spec,
    ".pth": _read_torch_spec,
}

SPEC_WRITERS: dict[str, Callable[[Path, dict[str, torch.Tensor], str], None]] = {
    ".safetensors": _write_safetensors_spec,
    ".pt": _write_torch_spec,
    ".pth": _write_torch_spec,
}


def _require(path: Path, registry: Mapping[str, Any], action: str) -> Callable:
    """Look a suffix up in a registry, raising with the alternatives.

    Args:
        path: File whose suffix selects the handler.
        registry: Handlers accepted at this position.
        action: What the caller was attempting, for the error message.

    Raises:
        ValueError: If the suffix names no known format.
    """
    handler = registry.get(path.suffix)
    if handler is None:
        raise ValueError(
            f"Cannot {action} '{path.suffix}'; choose from {sorted(registry)}."
        )
    return handler


def reader_for(path: Path) -> Callable[[Path], dict[str, torch.Tensor]]:
    """Return the reader for a path's suffix.

    Exposed alongside `read_state` so a caller can reject a path before doing
    work that only one process should do.

    Args:
        path: File whose suffix selects the reader.
    """
    return _require(path, READERS, "read")


def writer_for(path: Path) -> Callable[[Path, dict[str, torch.Tensor]], None]:
    """Return the writer for a path's suffix.

    Args:
        path: File whose suffix selects the writer.
    """
    return _require(path, WRITERS, "write")


def read_state(path: Path) -> dict[str, torch.Tensor]:
    """Read a state dict, picking the reader from the suffix.

    Args:
        path: File to read.
    """
    reader = reader_for(path)
    return reader(path)


def write_state(
    path: Path, state: dict[str, torch.Tensor], spec: ModelSpec | None = None
) -> None:
    """Write a state dict, picking the writer from the suffix.

    A spec travels in the file's own metadata, so the weights carry what it
    takes to rebuild the model rather than relying on the caller to remember.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: How to rebuild the model. Safetensors keeps it in the file's
            metadata header; a torch archive keeps it under a reserved key
            beside the tensors, which the reader strips out again.

    Raises:
        ValueError: If the suffix names no known format.
    """
    if spec is None:
        writer_for(path)(path, state)
        return
    _require(path, SPEC_WRITERS, "write")(path, state, spec.to_json())


def read_spec(path: Path) -> ModelSpec | None:
    """Return the spec a file carries, or `None` if it carries none.

    Args:
        path: File to read.
    """
    stored = _require(path, SPEC_READERS, "read")(path)
    return ModelSpec.from_json(stored) if stored else None


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
    spec = read_spec(path)
    if spec is None:
        raise ValueError(
            f"No model spec in {str(path)!r}; weights alone do not say which "
            "architecture built them. Save it with write_state(..., spec=...)."
        )
    model = spec.build(**overrides)
    model.load_state_dict(read_state(path), strict=strict)
    return model

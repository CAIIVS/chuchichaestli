# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading named values out of a batch, whatever shape the loader returns."""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import torch

from chuchichaestli.utils.functools import map_nested
from chuchichaestli.utils.tensors import as_inexact, sanitize_ndim


__all__ = [
    "BatchType",
    "IMAGE_CHANNELS",
    "as_image_batch",
    "batch_to_device",
    "input_in_batch",
    "samples_in_batch",
    "unpack_batch",
]


BatchType = torch.Tensor | Mapping[str, Any] | Sequence[Any]

IMAGE_CHANNELS = frozenset({1, 3, 4})


def as_image_batch(images: Any) -> torch.Tensor:
    """Return images as an `(N, C, H, W)` float tensor on the host.

    Args:
        images: A tensor, or anything `torch.as_tensor` accepts,
            laid out channels-first or channels-last.

    Raises:
        ValueError: If the shape is neither an image's nor a batch of them.
    """
    x = as_inexact(torch.as_tensor(images).detach().cpu())
    if x.ndim == 3 and not IMAGE_CHANNELS & {x.shape[0], x.shape[-1]}:
        x = x[:, None]
    x = sanitize_ndim(x)
    if x.shape[1] in IMAGE_CHANNELS:
        return x
    if x.shape[-1] in IMAGE_CHANNELS:
        return x.permute(0, 3, 1, 2)
    raise ValueError(
        f"A batch of images needs 1, 3 or 4 channels first or last, "
        f"got shape {tuple(x.shape)}."
    )


def unpack_batch(
    batch: BatchType, *names: str, reader: str = "Batch reader"
) -> tuple[Any, ...]:
    """Return the named values of a batch, in the order asked for.

    A mapping is read by name and a sequence by position, which are the two
    shapes `FileDataset.return_as` produces.

    Args:
        batch: A mapping, or a sequence holding one value per name.
        names: Keys to read, e.g. `"x"` and `"y"`.
        reader: What to call the caller in an error message.

    Raises:
        ValueError: If no name was asked for, or the batch holds no such
            values.
    """
    if not names:
        raise ValueError("Unpacking a batch needs at least one name.")
    if isinstance(batch, Mapping):
        missing = sorted(set(names) - set(batch))
        if missing:
            raise ValueError(
                f"Batch has no {missing} to read; it holds {sorted(batch)}."
            )
        return tuple(batch[name] for name in names)
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if len(batch) != len(names):
            raise ValueError(
                f"{reader} reads {len(names)} values, got {len(batch)} items."
            )
        return tuple(batch)
    raise ValueError(
        f"{reader} reads {len(names)} values, got a {type(batch).__name__}."
    )


def samples_in_batch(batch: BatchType) -> int:
    """Return how many samples a batch holds.

    Args:
        batch: A tensor, or a mapping or sequence holding one.
    """
    if isinstance(batch, torch.Tensor):
        return len(batch)
    if isinstance(batch, Mapping):
        candidates: Iterable[Any] = batch.values()
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        candidates = batch
    else:
        return 1
    for value in candidates:
        if isinstance(value, torch.Tensor):
            return len(value)
    return 1


def batch_to_device(batch: BatchType, device: torch.device | str) -> BatchType:
    """Return a batch with every tensor in it on a device.

    Args:
        batch: A batch as the loader produced it.
        device: Where its tensors should land.
    """
    return map_nested(batch, lambda tensor: tensor.to(device, non_blocking=True))


def input_in_batch(batch: BatchType, name: str = "x") -> Any:
    """Return the one value a batch leads with.

    Unlike `unpack_batch` this ignores whatever else the batch carries, so a
    batch holding a target alongside its input still yields the input.

    Args:
        batch: A mapping, a sequence, or a lone tensor.
        name: Key the value is read from, for mapping batches.

    Raises:
        ValueError: If a mapping does not hold the name.
    """
    if isinstance(batch, Mapping):
        if name not in batch:
            raise ValueError(
                f"Batch has no {name!r} to read; it holds {sorted(batch)}."
            )
        return batch[name]
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if not batch:
            raise ValueError("Batch is empty, so it leads with nothing.")
        return batch[0]
    return batch

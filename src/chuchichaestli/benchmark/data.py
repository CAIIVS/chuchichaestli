# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""What every benchmark reading a dataset needs.

Whatever a data benchmark compares, it measures the same thing: a `DataLoader`
walking the dataset once. `read_epoch` is that walk, over any case that says how
the loader draws (`DataLoaderCase`), and `drop_page_cache` is what makes the
read a cold one.

```python
    drop_page_cache(path)
    dataset = MyDataset(path)
    read_epoch(dataset, case, args.device)
```

What a sweep varies -- how many samples, which sizes, how many workers -- is the
benchmark's own, and belongs in the benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset

from chuchichaestli.utils import map_nested


__all__ = ["DataLoaderCase", "drop_page_cache", "read_epoch"]


def drop_page_cache(path: str | Path) -> None:
    """Drop a file's pages, so the next read of it has to go to storage.

    Best effort, which is all a benchmark gets without being root:
    `POSIX_FADV_DONTNEED` drops clean pages, so it reaches neither a file some
    process still has memory-mapped nor anything living on `tmpfs`.

    Args:
        path: File to evict.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    handle = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(handle)


@runtime_checkable
class DataLoaderCase(Protocol):
    """A case that says how a loader draws from a dataset."""

    batch_size: int
    workers: int
    order: str
    seed: int


def read_epoch(
    dataset: Dataset, case: DataLoaderCase, device: torch.device | str = "cpu"
) -> None:
    """Walk the dataset once through a loader, the way an epoch of training does.

    Every tensor of a batch is landed on the device, whether the dataset yields
    one, a tuple, or a dict of them.

    Args:
        dataset: Dataset to read.
        case: Case being read.
        device: Device the batches are landed on.
    """
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=case.batch_size,
        shuffle=case.order == "shuffled",
        num_workers=case.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(case.seed),
    )
    if device.type == "cpu":
        for _ in loader:
            pass
        return
    for batch in loader:
        map_nested(batch, lambda x: x.to(device, non_blocking=True))
    torch.cuda.synchronize()

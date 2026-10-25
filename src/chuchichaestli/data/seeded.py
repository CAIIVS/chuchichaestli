# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batches loaded under seeds of their own, whichever worker loads them."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from torch.utils.data import Dataset
from chuchichaestli.utils.rng import derive_seed, draw_seed, scoped_seed


__all__ = ["SeededBatches"]


class SeededBatches(Dataset):
    """A dataset loading each batch under a seed derived from the epoch and its indices."""

    def __init__(self, dataset: Dataset, seed: int | None = None):
        """Constructor.

        Args:
            dataset: What the samples come from.
            seed: Seed each batch's own derives from, with the epoch and its indices.
        """
        self.dataset = dataset
        self.seed = draw_seed(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the next batches derive their seeds from.

        Args:
            epoch: The pass about to start.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        """Return one sample, loaded as a batch of its own.

        Args:
            index: Which sample.
        """
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:
        """Return a batch's samples, their randomness drawn from the epoch and indices.

        Args:
            indices: The samples in the batch.
        """
        key = f"epoch={self.epoch}/batch={','.join(map(str, indices))}"
        with scoped_seed(derive_seed(self.seed, key)):
            fetch = getattr(self.dataset, "__getitems__", None)
            if fetch is not None:
                return list(fetch(indices))
            return [self.dataset[index] for index in indices]

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for batches loaded under seeds of their own."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from chuchichaestli.data import SeededBatches


class Jittered(Dataset):
    """A dataset jittering each sample with torch's, numpy's and python's generators."""

    def __init__(self, n: int = 12):
        """Constructor.

        Args:
            n: How many samples.
        """
        self.n = n

    def __len__(self) -> int:
        """Number of samples."""
        return self.n

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the index beside a draw from each global generator.

        Args:
            index: Which sample.
        """
        draws = [float(torch.rand(())), float(np.random.rand()), random.random()]
        return torch.tensor([index, *draws], dtype=torch.float64)


class Batched(Jittered):
    """A jittered dataset that loads a whole batch at once."""

    def __init__(self, n: int = 12):
        """Constructor.

        Args:
            n: How many samples.
        """
        super().__init__(n)
        self.calls: list[list[int]] = []

    def __getitems__(self, indices: list[int]) -> list[torch.Tensor]:
        """Return a batch's samples, noting that it was asked for.

        Args:
            indices: The samples in the batch.
        """
        self.calls.append(list(indices))
        return [self[index] for index in indices]


def batch(dataset: SeededBatches, indices: list[int]) -> list[list[float]]:
    """Return a batch's samples as lists.

    Args:
        dataset: What to load from.
        indices: The samples in the batch.
    """
    return [sample.tolist() for sample in dataset.__getitems__(indices)]


def test_the_same_batch_draws_the_same():
    """A batch's draws follow from the seed, the epoch and its indices alone."""
    seeded, other = SeededBatches(Jittered(), seed=3), SeededBatches(Jittered(), seed=4)
    assert batch(seeded, [1, 2, 3]) == batch(seeded, [1, 2, 3])
    assert batch(seeded, [1, 2, 3]) != batch(other, [1, 2, 3])


def test_each_epoch_draws_anew():
    """set_epoch moves on to fresh draws, and back to the old ones."""
    seeded = SeededBatches(Jittered(), seed=3)
    first = batch(seeded, [1, 2, 3])
    seeded.set_epoch(1)
    assert batch(seeded, [1, 2, 3]) != first
    seeded.set_epoch(0)
    assert batch(seeded, [1, 2, 3]) == first


def test_the_draws_do_not_depend_on_the_worker_count():
    """With a plain DataLoader, who loads a batch does not matter."""

    def loaded(**kwargs):
        loader = DataLoader(SeededBatches(Jittered(), seed=3), batch_size=4, **kwargs)
        return [b.tolist() for b in loader]

    assert loaded(num_workers=2, multiprocessing_context="spawn") == loaded()


def test_an_unseeded_wrapper_draws_its_seed_from_torch():
    """Two unseeded wrappers differ, and repeat once torch is seeded again."""
    torch.manual_seed(0)
    first, second = SeededBatches(Jittered()), SeededBatches(Jittered())
    torch.manual_seed(0)
    assert first.seed != second.seed
    assert SeededBatches(Jittered()).seed == first.seed


def test_a_batch_is_loaded_the_way_the_dataset_loads_one():
    """A dataset's own batched loading is kept, under the batch's seed."""
    dataset = Batched()
    seeded = SeededBatches(dataset, seed=3)
    batch(seeded, [4, 5])
    assert dataset.calls == [[4, 5]]
    assert batch(seeded, [4, 5]) == batch(SeededBatches(Jittered(), seed=3), [4, 5])


def test_loading_leaves_the_global_streams_alone():
    """The caller's draws must not move with the data."""

    def draws():
        return torch.rand(2).tolist(), np.random.rand(2).tolist(), random.random()

    seeded = SeededBatches(Jittered(), seed=3)
    torch.manual_seed(0), np.random.seed(0), random.seed(0)
    expected = draws()
    torch.manual_seed(0), np.random.seed(0), random.seed(0)
    batch(seeded, [1, 2, 3])
    seeded[5]
    assert draws() == expected

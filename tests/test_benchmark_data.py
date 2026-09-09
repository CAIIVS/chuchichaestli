# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for what the benchmarks reading a dataset share."""

import os
from dataclasses import dataclass

import pytest
import torch
from chuchichaestli.benchmark.data import DataLoaderCase, drop_page_cache, read_epoch


@dataclass(frozen=True)
class Case:
    """A case owing nothing to `TensorCase`, to keep the protocol structural."""

    batch_size: int = 4
    workers: int = 0
    order: str = "sequential"
    seed: int = 0


class Counted(torch.utils.data.Dataset):
    """A dataset counting what a loader drew from it."""

    def __init__(self, samples: int, shape: tuple[int, ...] = (2,), wrap=None):
        """Hold this many samples of this shape.

        Args:
            samples: Samples the dataset holds.
            shape: Shape of one sample.
            wrap: Returns what one sample looks like, given its tensor; the
                bare tensor if omitted.
        """
        self.samples = samples
        self.shape = shape
        self.wrap = wrap or (lambda item: item)
        self.drawn: list[int] = []

    def __len__(self) -> int:
        """Number of samples the dataset holds."""
        return self.samples

    def __getitem__(self, index: int):
        """Return one sample, and remember it was asked for.

        Args:
            index: Sample to draw.
        """
        self.drawn.append(index)
        return self.wrap(torch.zeros(self.shape))


class TestDropPageCache:
    """Evicting a file, so the next read of it is a cold one."""

    def test_leaves_the_file_alone(self, tmp_path):
        """Only the pages go; what a benchmark then reads is unchanged."""
        path = tmp_path / "sample.bin"
        path.write_bytes(b"chuchichaestli" * 1024)
        drop_page_cache(path)
        assert path.read_bytes() == b"chuchichaestli" * 1024

    def test_accepts_a_string(self, tmp_path):
        """A path is a `str` as readily as a `Path`."""
        path = tmp_path / "sample.bin"
        path.write_bytes(b"x")
        drop_page_cache(str(path))

    def test_is_a_no_op_without_fadvise(self, tmp_path, monkeypatch):
        """A platform without `posix_fadvise` gets no error, just no eviction."""
        path = tmp_path / "sample.bin"
        path.write_bytes(b"x")
        monkeypatch.delattr(os, "posix_fadvise", raising=False)
        drop_page_cache(path)


class TestDataLoaderCase:
    """What a case has to say for an epoch to be read over it."""

    def test_is_structural(self):
        """Anything carrying the four loader knobs is a case, base class aside."""
        assert isinstance(Case(), DataLoaderCase)

    def test_a_case_short_of_one_is_not(self):
        """A case missing a knob `read_epoch` reads is not mistaken for one."""

        @dataclass(frozen=True)
        class Partial:
            batch_size: int = 4

        assert not isinstance(Partial(), DataLoaderCase)


class TestReadEpoch:
    """Walking a dataset once, the way an epoch of training does."""

    def test_draws_every_sample_once(self):
        """An epoch is the whole dataset, whatever the batch does not divide."""
        dataset = Counted(10)
        read_epoch(dataset, Case(batch_size=4))
        assert sorted(dataset.drawn) == list(range(10))

    def test_sequential_draws_in_order(self):
        """A sequential case reads the dataset front to back."""
        dataset = Counted(8)
        read_epoch(dataset, Case(batch_size=2))
        assert dataset.drawn == list(range(8))

    def test_shuffled_is_seeded(self):
        """A shuffled case draws in the same order every run, and not in order."""
        case = Case(batch_size=4, order="shuffled")
        runs = []
        for _ in range(2):
            dataset = Counted(32)
            read_epoch(dataset, case)
            runs.append(dataset.drawn)
        assert runs[0] == runs[1]
        assert runs[0] != list(range(32))

    def test_seed_picks_the_order(self):
        """Two seeds draw a shuffled dataset in two different orders."""
        orders = []
        for seed in (0, 1):
            dataset = Counted(32)
            read_epoch(dataset, Case(batch_size=4, order="shuffled", seed=seed))
            orders.append(dataset.drawn)
        assert orders[0] != orders[1]

    def test_device_may_be_a_string(self):
        """A benchmark that keeps `args.device` around need not wrap it."""
        dataset = Counted(4)
        read_epoch(dataset, Case(batch_size=2), "cpu")
        assert sorted(dataset.drawn) == list(range(4))

    @pytest.mark.parametrize(
        "wrap",
        [
            None,
            lambda item: (item, torch.ones(1)),
            lambda item: {"data": item, "attrs": torch.ones(1)},
            lambda item: [item, [item]],
        ],
    )
    def test_reads_whatever_a_sample_looks_like(self, wrap):
        """A dataset yields one tensor, a tuple, or a dict; an epoch is an epoch."""
        dataset = Counted(8, wrap=wrap)
        read_epoch(dataset, Case(batch_size=2))
        assert sorted(dataset.drawn) == list(range(8))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a gpu")
    @pytest.mark.parametrize(
        "wrap",
        [
            None,
            lambda item: (item, torch.ones(1)),
            lambda item: {"data": item, "attrs": torch.ones(1)},
        ],
    )
    def test_lands_every_tensor_of_a_batch(self, wrap):
        """On a gpu, a tuple or dict of tensors is landed, not just a bare one."""
        dataset = Counted(8, wrap=wrap)
        read_epoch(dataset, Case(batch_size=2), "cuda")
        assert sorted(dataset.drawn) == list(range(8))

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resumable, deterministically ordered passes over a dataset."""

from __future__ import annotations
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from multiprocessing.context import BaseContext
from typing import Any
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from chuchichaestli.data.batch import BatchType
from chuchichaestli.data.split import split_dataset
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.traits import Topology
from chuchichaestli.utils.rng import rng_generator


__all__ = ["DataManager", "C3liDataError"]


class C3liDataError(ValueError):
    """Raised when batches cannot be drawn in a reproducible order."""


class DataManager:
    """Supplies a dataset's batches in an order derived from the seed.

    The unit ordered, seeked and sharded is a batch, so a batch size and a
    sampler's variable batches take the same path.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        *,
        batches: Sampler[list[int]] | Sequence[Sequence[int]] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        collate_fn: Callable[[Sequence[Any]], BatchType] | None = None,
        pin_memory: bool = False,
        pin_memory_device: str = "",
        timeout: float = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        multiprocessing_context: str | BaseContext | None = None,
        seed: int | None = None,
        split: Sequence[float] | Mapping[str, float] | None = None,
        part: str | int | None = None,
        split_seed: int = 0,
        balance_ranks: bool = False,
    ):
        """Constructor.

        Args:
            dataset: What the batches index into.
            batch_size: Samples per batch when no sampler is given.
            batches: A batch sampler or an explicit list of index lists,
                for variable-size batches. Its contents are used as given.
            shuffle: Whether the batch order is permuted each epoch.
            drop_last: Whether a short final batch is discarded.
            num_workers: Dataloader worker processes.
            collate_fn: Builds a batch from the samples it indexes.
            pin_memory: Whether batches are copied into pinned memory.
            pin_memory_device: Device to pin into; the current one if empty.
            timeout: Seconds to wait for a batch from a worker; `0` waits.
            prefetch_factor: Batches each worker loads ahead.
            persistent_workers: Whether workers outlive one epoch.
            multiprocessing_context: How workers are started, e.g. `"spawn"`.
            seed: Root the loader's own generator derives from, epoch by
                epoch; taken from the context when absent.
            split: Shares to divide the dataset into, for drawing train and
                evaluation batches from one source. Every manager splitting
                the same dataset must pass the same shares and `split_seed`.
            part: Which share this manager draws from, named or indexed.
            split_seed: Seed the split derives from.
            balance_ranks: Whether a batch is weighed against every process's
                samples rather than this one's alone, which differs only when
                they hold unequal numbers.

        Raises:
            ValueError: If a number is out of range, if `split` and `part` are
                not given together, or if a worker option was given without
                workers to apply it to.
            C3liDataError: If a sampler was given that shuffles itself.
        """
        if batch_size < 1:
            raise ValueError(
                f"A data manager needs a positive batch size, got {batch_size!r}."
            )
        if num_workers < 0:
            raise ValueError(
                f"A data manager needs no negative workers, got {num_workers!r}."
            )
        if getattr(batches, "shuffle", False):
            raise C3liDataError(
                f"{type(batches).__name__} was given shuffle=True, which draws "
                "from the global generator and so would order batches "
                "differently on a resume. Construct it with shuffle=False and "
                "let DataManager(shuffle=...) do the ordering."
            )
        if timeout < 0:
            raise ValueError(
                f"A data manager needs no negative timeout, got {timeout!r}."
            )
        if not num_workers:
            for label, value in (
                ("prefetch_factor", prefetch_factor),
                ("persistent_workers", persistent_workers or None),
            ):
                if value is not None:
                    raise ValueError(
                        f"{label}={value!r} needs workers to apply to; this "
                        "manager reads in the main process."
                    )
        if (split is None) != (part is None):
            raise ValueError(
                f"A split needs a part to draw from: got {split=}, {part=}."
            )
        if split is not None:
            dataset = self._subset(dataset, split, part, split_seed)
        self.dataset = dataset
        self.split = split
        self.part = part
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.seed = seed
        self.balance_ranks = balance_ranks
        self._batches = (
            [list(batch) for batch in batches] if batches is not None else None
        )

    @classmethod
    def from_source(
        cls, source: DataManager | Dataset | None, **kwargs: Any
    ) -> DataManager:
        """Return a manager for whatever a caller was given.

        Args:
            source: A manager, or a dataset to build one around.
            kwargs: Passed to the constructor (for a dataset source).

        Raises:
            ValueError: If there is nothing to draw batches from, or options
                a built manager would ignore.
        """
        if isinstance(source, DataManager):
            if kwargs:
                raise ValueError(
                    f"{sorted(kwargs)} were given alongside a built "
                    "DataManager, which carries its own; set them in one place."
                )
            return source
        if source is None:
            raise ValueError("A loop needs data to draw batches from.")
        return cls(source, **kwargs)

    @staticmethod
    def _subset(
        dataset: Dataset,
        split: Sequence[float] | Mapping[str, float],
        part: str | int,
        seed: int,
    ) -> Dataset:
        """Return the part of a split dataset this manager draws from.

        Args:
            dataset: What to divide.
            split: Shares to divide it into.
            part: Which share to take.
            seed: Seed the split derives from.

        Raises:
            C3liDataError: If the split holds no such part.
        """
        parts = split_dataset(dataset, split, seed=seed)
        try:
            return parts[part]
        except (KeyError, IndexError):
            held = list(parts) if isinstance(parts, dict) else list(range(len(parts)))
            raise C3liDataError(
                f"The split has no part {part!r}; it holds {held}."
            ) from None

    def __repr__(self) -> str:
        """Return a short description of the manager."""
        batch_size, num_workers = self.batch_size, self.num_workers
        source = "batched" if self._batches is not None else "chunked"
        part = f", part={self.part!r}" if self.part is not None else ""
        return f"DataManager({source}, {batch_size=}, {num_workers=}{part})"

    def plan(self, ctx: Context, epoch: int = 0) -> list[list[int]]:
        """Return the batches of one epoch, in the order they are drawn.

        Args:
            ctx: Context the order derives from.
            epoch: Which pass over the dataset.
        """
        source = self._batches if self._batches is not None else self.dataset
        order = (
            RandomSampler(source, generator=ctx.rng(f"data/epoch={epoch}"))
            if self.shuffle
            else SequentialSampler(source)
        )
        if self._batches is not None:
            return [self._batches[position] for position in order]
        return list(BatchSampler(order, self.batch_size, self.drop_last))

    def shard(self, plan: list[list[int]], topology: Topology) -> list[list[int]]:
        """Return this process's share, with one batch count for every rank.

        Under a distributed topology an uneven share is a deadlock rather than
        a rounding error, so a short plan wraps from its own front.

        Args:
            plan: The epoch's batches.
            topology: Process layout to divide across.
        """
        if topology.world_size == 1:
            return plan
        if not plan:
            return []
        per_rank = math.ceil(len(plan) / topology.world_size)
        padded = plan + plan[: per_rank * topology.world_size - len(plan)]
        return padded[topology.rank :: topology.world_size]

    def sharded_plan(self, ctx: Context, epoch: int = 0) -> list[list[int]]:
        """Return the batches this process draws in one epoch.

        Args:
            ctx: Context the order derives from.
            epoch: Which pass over the dataset.
        """
        return self.shard(self.plan(ctx, epoch), ctx.topology)

    def iter(self, ctx: Context, epoch: int = 0, seek: int = 0) -> Iterator[BatchType]:
        """Iterate this process's batches for one epoch.

        Args:
            ctx: Context the order and the worker seeds derive from.
            epoch: Which pass over the dataset.
            seek: Batches to skip, for resuming part-way through an epoch.
        """
        plan = self.sharded_plan(ctx, epoch)[seek:]
        if not plan:
            return iter(())
        key = f"data/epoch={epoch}"
        loader = DataLoader(
            self.dataset,
            batch_sampler=plan,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=ctx.seeder(key) if self.num_workers else None,
            generator=self._generator(ctx, key),
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
            timeout=self.timeout,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.persistent_workers,
            **(
                {"prefetch_factor": self.prefetch_factor}
                if self.prefetch_factor is not None
                else {}
            ),
        )
        return iter(loader)

    def _generator(self, ctx: Context, key: str) -> torch.Generator:
        """Return the generator the loader seeds its workers from.

        Args:
            ctx: Context the seed derives from when none was pinned.
            key: Names the random stream within the run.
        """
        if self.seed is None:
            return ctx.rng(key)
        return rng_generator(self.seed, key)

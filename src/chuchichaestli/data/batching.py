# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch batch sampler for hierarchical or grouped batching."""

import re
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence, Sized
import torch
from torch.utils.data import Sampler
from chuchichaestli.data.base import FileDataset


class HierarchicalBatchSampler(Sampler):
    """Batch sampler that operates on pre-built hierarchical batch indices.

    Accepts a list of batches (each a list of integer indices) directly, making
    it useful when the grouping logic is handled externally or when subclassing.
    """

    def __init__(self, batches: list[list[int]], shuffle: bool = False):
        """Constructor.

        Args:
            batches: Pre-built list of batches; each batch is a list of indices.
            shuffle: Shuffle the order of batches each epoch (internal batch
                ordering always preserved).
        """
        self._batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        """Iterator."""
        if self.shuffle:
            # Shuffle via torch's RNG
            order = torch.randperm(len(self._batches)).tolist()
            yield from (self._batches[i] for i in order)
        else:
            yield from self._batches

    def __len__(self) -> int:
        """Number of batches (not samples)."""
        return len(self._batches)

    @staticmethod
    def batch_indices(*args, drop_last: bool = False) -> list[list[int]]:
        """Partition indices from `start` to (excluding) `end` into batches.

        Args:
            *args: Starting index, stop index, and size of batches; can be
                (start, end, size) or (end, size) with start=0 or (end,) with start=0, size=1.
            drop_last: Drop the last batch if its size is smaller.
        """
        match args:
            case (end,):
                start, size = 0, 1
            case (end, size):
                start = 0
            case (start, end, size):
                pass
            case _:
                raise TypeError(
                    f"`batch_indices` takes 1 to 3 positional arguments, got {len(args)}."
                )
        indices = range(start, end)
        batches = [list(indices[i : i + size]) for i in range(0, len(indices), size)]
        if drop_last and batches and len(batches[-1]) < size:
            batches = batches[:-1]
        return batches


class HierarchicalFileBatchSampler(HierarchicalBatchSampler):
    """Batches samples grouped by a hierarchy key derived from file paths.

    Files sharing a `key_fn(path)` form a group; `order_fn` optionally sorts
    within a group. `batch_size` partitions each group into fixed-size batches,
    or `None` merges a whole group into one variable-length batch. See the
    `from_filename`, `from_directory`, and `from_sequences` factories.
    """

    def __init__(
        self,
        dataset: FileDataset,
        key_fn: Callable[[Path], str],
        batch_size: int | None = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        order_fn: Callable[[Path], Any] | None = None,
    ):
        """Constructor.

        Args:
            dataset: Any `FileDataset` subclass.
            key_fn: Maps a `Path` to group label string.
            batch_size: Samples per batch. The sentinel `None` merges each
                group into a single variable-length batch.
            shuffle: Shuffle batch order each epoch; batch contents are preserved.
            drop_last: Drop partial batches at the end of each group. No-op when
                `batch_size is None`.
            order_fn: Optional `Path -> sortable` key used to order files within
                a group before emitting their sample indices. Default `None`
                preserves dataset/glob order.
        """
        if not hasattr(dataset, "files"):
            raise TypeError(
                "Dataset must be a FileDataset subclass or at least have the 'files' list attribute."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.key_fn = key_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.order_fn = order_fn
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        """Group batches with the key function based on dataset filenames."""
        offsets = self.dataset._file_offsets
        files = self.dataset.files

        # Group file indices by key, preserving first-seen key order.
        groups: dict[str, list[int]] = {}
        for file_idx, path in enumerate(files):
            key = self.key_fn(path)
            groups.setdefault(key, []).append(file_idx)

        batches: list[list[int]] = []
        for file_indices in groups.values():
            if self.order_fn is not None:
                file_indices = sorted(
                    file_indices, key=lambda i: self.order_fn(files[i])
                )
            if self.batch_size is None:
                # One variable-length batch per group (whole sequence merged in
                # order). drop_last is a documented no-op in this mode.
                merged = [
                    idx
                    for file_idx in file_indices
                    for idx in range(offsets[file_idx], offsets[file_idx + 1])
                ]
                if merged:
                    batches.append(merged)
            else:
                for file_idx in file_indices:
                    batches.extend(
                        self.batch_indices(
                            offsets[file_idx],
                            offsets[file_idx + 1],
                            self.batch_size,
                            drop_last=self.drop_last,
                        )
                    )

        return batches

    @classmethod
    def from_filename(
        cls,
        dataset: FileDataset,
        pattern: str,
        group: int = 1,
        fallback: str = "__ungrouped__",
        batch_size: int | None = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        order_fn: Callable[[Path], Any] | None = None,
    ) -> "HierarchicalBatchSampler":
        """Group files by a capture group in their filename.

        Args:
            dataset: Any `FileDataset` subclass.
            pattern: Regex applied to the filename (`path.name`, including the
                suffix) of each dataset file.
            group: Which capture group to use as the key (default: 1).
            fallback: Key used when the pattern does not match.
            batch_size: Samples per batch (sentinel `None` merges each group).
            shuffle: Shuffle batch order each epoch; batch contents are preserved.
            drop_last: Drop partial batches at the end of each group.
            order_fn: Optional `Path -> sortable` ordering within each group.
        """
        rx = re.compile(pattern)

        def key_fn(path: Path) -> str:
            m = rx.search(path.name)
            return m.group(group) if m else fallback

        return cls(
            dataset,
            key_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            order_fn=order_fn,
        )

    @classmethod
    def from_sequences(
        cls,
        dataset: FileDataset,
        pattern: str,
        group: int = 1,
        order_group: int = 2,
        order_cast: Callable[[str], Any] = int,
        fallback: str = "__ungrouped__",
        shuffle: bool = False,
    ) -> "HierarchicalBatchSampler":
        """Emit one variable-length, order-sorted batch per filename-key group.

        Each batch is a group's samples in `order_group` order (e.g. one full
        simulation trajectory).

        Args:
            dataset: Any `FileDataset` subclass.
            pattern: Regex exposing both `group` and `order_group` captures.
            group: Capture group used as the sequence key (default: 1).
            order_group: Capture group ordering within a sequence (default: 2).
            order_cast: Cast for the order capture into a sortable (default: `int`).
            fallback: Key for unmatched files (merged into one batch; warns).
            shuffle: Shuffle sequence order; intra-sequence order is preserved.
        """
        rx = re.compile(pattern)

        def key_fn(path: Path) -> str:
            m = rx.search(path.name)
            return m.group(group) if m else fallback

        def order_fn(path: Path) -> Any:
            m = rx.search(path.name)
            if m is None:
                return order_cast("0")
            return order_cast(m.group(order_group))

        unmatched = [p for p in dataset.files if rx.search(p.name) is None]
        if unmatched:
            warnings.warn(
                f"{len(unmatched)} file(s) did not match pattern {pattern!r} and "
                f"will be merged into a single '{fallback}' sequence batch "
                f"(e.g. {unmatched[0].name!r}).",
                UserWarning,
                stacklevel=2,
            )

        return cls(
            dataset,
            key_fn,
            batch_size=None,
            shuffle=shuffle,
            order_fn=order_fn,
        )

    @classmethod
    def from_directory(
        cls,
        dataset: FileDataset,
        depth: int = 1,
        batch_size: int | None = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        order_fn: Callable[[Path], Any] | None = None,
    ) -> "HierarchicalBatchSampler":
        """Group files by the directory `depth` levels above each file.

        Args:
            dataset: Any `FileDataset` subclass.
            depth: directory depth above a file; default is 1, i.e. the
                immediate parent directory name.
            batch_size: Samples per batch (sentinel `None` merges each group).
            shuffle: Shuffle batch order each epoch; batch contents are preserved.
            drop_last: Drop partial batches at the end of each group.
            order_fn: Optional `Path -> sortable` ordering within each group.
        """

        def key_fn(path: Path) -> str:
            parts = path.parts
            idx = -(depth + 1)  # depth=0 is immediate parent
            return parts[idx] if abs(idx) <= len(parts) else parts[0]

        return cls(
            dataset,
            key_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            order_fn=order_fn,
        )


class SlidingWindowBatchSampler(HierarchicalBatchSampler):
    """Batches sliding windows of consecutive samples over ordered sequences.

    Each emitted batch is `batch_size` windows of size `window_size` (optionally
    trailed by `horizon`), **flattened** into one index list of length
    `batch_size * (window_size + horizon)`.
    """

    def __init__(
        self,
        dataset: Sized,
        window_size: int,
        horizon: int = 0,
        stride: int = 1,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        boundaries: Sequence[int] | None = None,
    ):
        """Constructor.

        Args:
            dataset: Any sized dataset. If it exposes `_file_offsets` (any
                `FileDataset`), those are used as sequence boundaries by default.
            window_size: Number of consecutive samples forming the input window.
            horizon: Number of further consecutive samples appended to each
                window as the prediction target. `0` yields windows only.
            stride: Step between successive window starts. `stride < window_size`
                produces overlapping windows.
            batch_size: Windows per batch (not samples per batch).
            shuffle: Shuffle batch order each epoch; batch contents are preserved.
            drop_last: Drop a trailing batch holding fewer than `batch_size`
                windows. Windows themselves are always full by construction.
            boundaries: Cumulative offsets delimiting independent sequences, e.g.
                `[0, n0, n0+n1, ...]`. Defaults to the dataset's `_file_offsets`,
                falling back to a single sequence spanning the whole dataset.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.boundaries = self._resolve_boundaries(dataset, boundaries)
        self._starts = self._build_starts()

    @property
    def span(self) -> int:
        """Number of samples in one window, including the horizon."""
        return self.window_size + self.horizon

    @property
    def n_windows(self) -> int:
        """Total number of windows across all sequences."""
        n = len(self._starts)
        if self.drop_last:
            n -= n % self.batch_size
        return n

    @staticmethod
    def _resolve_boundaries(
        dataset: Sized, boundaries: Sequence[int] | None
    ) -> list[int]:
        """Pick the sequence boundaries, validating whatever is supplied."""
        n = len(dataset)
        if boundaries is None:
            offsets = getattr(dataset, "_file_offsets", None)
            # `_file_offsets` can be empty or stale
            if offsets and offsets[0] == 0 and offsets[-1] == n:
                return list(offsets)
            return [0, n]
        boundaries = list(boundaries)
        if len(boundaries) < 2:
            raise ValueError(
                f"boundaries needs at least a start and an end, got {boundaries}"
            )
        if any(a > b for a, b in zip(boundaries[:-1], boundaries[1:], strict=False)):
            raise ValueError(f"boundaries must be non-decreasing, got {boundaries}")
        if boundaries[0] < 0 or boundaries[-1] > n:
            raise ValueError(f"boundaries must lie within [0, {n}], got {boundaries}")
        return boundaries

    def _build_starts(self) -> list[int]:
        """Collect the start index of every window, in sequence order.

        Only the starts are kept; a batch is expanded to its flat index list
        on demand in `__iter__`, so peak memory stays proportional to the
        window count rather than to `n_windows * span`.
        """
        starts: list[int] = []
        for lo, hi in zip(self.boundaries[:-1], self.boundaries[1:], strict=False):
            # Last start for which a full window+horizon still fits the sequence.
            starts.extend(range(lo, hi - self.span + 1, self.stride))
        return starts

    def _batch(self, i: int) -> list[int]:
        """Expand the `i`-th batch of window starts into flat sample indices."""
        span = self.span
        chunk = self._starts[i * self.batch_size : (i + 1) * self.batch_size]
        return [idx for start in chunk for idx in range(start, start + span)]

    def __iter__(self):
        """Iterate batches, expanding one batch of windows at a time."""
        n = len(self)
        order = torch.randperm(n).tolist() if self.shuffle else range(n)
        yield from (self._batch(i) for i in order)

    def __len__(self) -> int:
        """Number of batches (not windows)."""
        n = len(self._starts)
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)

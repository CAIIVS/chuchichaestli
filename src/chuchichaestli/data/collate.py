# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collation utilities for sequence/loop batches."""

from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
from torch.utils.data import default_collate
from chuchichaestli.data.base import IndexedSample


__all__ = ["SequenceCollate", "sequence_collate"]


def _first_leaf(sample: Any) -> Any:
    """Return the first tensor-like leaf of a nested dict/tuple/list sample."""
    if isinstance(sample, dict):
        return _first_leaf(next(iter(sample.values())))
    if isinstance(sample, (tuple, list)):
        return _first_leaf(sample[0])
    return sample


class SequenceCollate:
    """Pickleable `collate_fn` that applies one shared transform per batch.

    Params are sampled once (via `param_fn`) and the same `transform(leaf,
    params)` is applied to every tensor leaf of every item, identically across
    all steps and paired fields, before `default_collate`. A class, not a
    closure, so it survives pickling for `DataLoader(num_workers>0)`.
    """

    def __init__(
        self,
        transform: Callable[[Any, Any], Any] | None = None,
        param_fn: Callable[[Any], Any] | None = None,
        source: Any | None = None,
        key_fn: Callable[[Path], str] | None = None,
    ):
        """Constructor.

        Args:
            transform: Callable `(leaf, params) -> leaf` applied to every leaf.
                `None` disables transforms (plain `default_collate`).
            param_fn: Callable `(first_leaf) -> params` sampling shared params
                once per batch. `None` passes `params=None` to `transform`.
            source: Optional `FileDataset` whose `.files`/`_file_offsets` are
                snapshotted (never the live dataset) so batches from
                `with_indices` carry `key`/`files`/`indices`.
            key_fn: Optional `Path -> str` labelling a batch's key from its first
                file. Must be picklable for `num_workers>0`.
        """
        self.transform = transform
        self.param_fn = param_fn
        self.key_fn = key_fn
        if source is not None:
            # Snapshot only picklable primitives, never the live mmap-holding
            # dataset, so workers stay lightweight and picklable.
            self.files: list[Path] | None = list(source.files)
            self.file_offsets: list[int] | None = list(source._file_offsets)
            self.new_axis: bool = bool(getattr(source, "new_axis", False))
        else:
            self.files = None
            self.file_offsets = None
            self.new_axis = False

    def _apply(self, sample: Any, params: Any) -> Any:
        """Apply `transform(leaf, params)` to every tensor leaf of `sample`."""
        if isinstance(sample, dict):
            return {k: self._apply(v, params) for k, v in sample.items()}
        if isinstance(sample, (tuple, list)):
            return type(sample)(self._apply(v, params) for v in sample)
        return self.transform(sample, params)

    def _file_of(self, index: int) -> Path:
        """Resolve a global sample index to its source `Path`."""
        assert self.files is not None
        if self.new_axis:
            return self.files[index]
        offsets = self.file_offsets or []
        for file_idx, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            if lo <= index < hi:
                return self.files[file_idx]
        raise IndexError(f"Index {index} out of range for source files")

    def __call__(self, samples: Sequence[Any]) -> Any:
        """Collate one batch, unwrapping `IndexedSample` pairs if present."""
        indices: list[int] | None = None
        if samples and isinstance(samples[0], IndexedSample):
            indices = [s.index for s in samples]
            samples = [s.sample for s in samples]

        if self.transform is not None and samples:
            params = (
                self.param_fn(_first_leaf(samples[0]))
                if self.param_fn is not None
                else None
            )
            samples = [self._apply(s, params) for s in samples]

        batch = default_collate(list(samples))

        if indices is not None and self.files is not None:
            files = [self._file_of(i) for i in indices]
            key = self.key_fn(files[0]) if (self.key_fn and files) else None
            provenance = {"key": key, "files": files, "indices": list(indices)}
            if isinstance(batch, dict):
                batch = {**batch, **provenance}
            else:
                batch = {"data": batch, **provenance}
        return batch


def sequence_collate(
    transform: Callable[[Any, Any], Any] | None = None,
    param_fn: Callable[[Any], Any] | None = None,
    source: Any | None = None,
    key_fn: Callable[[Path], str] | None = None,
) -> SequenceCollate:
    """Return a `SequenceCollate` instance (see `SequenceCollate`)."""
    return SequenceCollate(
        transform=transform, param_fn=param_fn, source=source, key_fn=key_fn
    )

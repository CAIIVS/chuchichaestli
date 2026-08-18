# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collation utilities for sequence/loop batches."""

from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
import torch
from torch.utils.data import default_collate
from torchvision.transforms.v2 import Transform
from chuchichaestli.data.base import IndexedSample


__all__ = ["SequenceCollate", "sequence_collate"]


def _flatten_leaves(sample: Any) -> list[Any]:
    """Return the flat list of leaves of a nested dict/tuple/list sample."""
    if isinstance(sample, dict):
        return [leaf for v in sample.values() for leaf in _flatten_leaves(v)]
    if isinstance(sample, tuple | list):
        return [leaf for v in sample for leaf in _flatten_leaves(v)]
    return [sample]


class SequenceCollate:
    """Pickleable `collate_fn` that applies one shared v2 transform per batch.

    A `torchvision.transforms.v2.Transform` has its params sampled once per batch
    (`make_params`) and its `transform(leaf, params)` is then applied to every
    tensor leaf of every item (identically across all steps and paired fields)
    before `default_collate`.
    """

    def __init__(
        self,
        transform: Transform | None = None,
        source: Any | None = None,
        key_fn: Callable[[Path], str] | None = None,
    ):
        """Constructor.

        Args:
            transform: A `torchvision.transforms.v2.Transform` applied with
                shared per-batch params. `None` disables transforms (plain
                `default_collate`).
            source: Optional `FileDataset` whose `.files`/`_file_offsets` are
                snapshotted (never the live dataset) so batches from
                `with_indices` carry `key`/`files`/`indices`.
            key_fn: Optional `Path -> str` labelling a batch's key from its first
                file. Must be picklable for `num_workers>0`.
        """
        self.transform = transform
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
        if isinstance(sample, tuple | list):
            return type(sample)(self._apply(v, params) for v in sample)
        if isinstance(sample, torch.Tensor):
            return self.transform.transform(sample, params)
        return sample

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
            params = self.transform.make_params(_flatten_leaves(samples[0]))
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
    transform: Transform | None = None,
    source: Any | None = None,
    key_fn: Callable[[Path], str] | None = None,
) -> SequenceCollate:
    """Return a `SequenceCollate` instance (see `SequenceCollate`)."""
    return SequenceCollate(transform=transform, source=source, key_fn=key_fn)

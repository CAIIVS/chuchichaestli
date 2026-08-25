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


__all__ = [
    "SequenceCollate",
    "sequence_collate",
]


def _flatten_leaves(sample: Any) -> list[Any]:
    """Return the flat list of leaves of a nested dict/tuple/list sample."""
    if isinstance(sample, dict):
        return [leaf for v in sample.values() for leaf in _flatten_leaves(v)]
    if isinstance(sample, tuple | list):
        return [leaf for v in sample for leaf in _flatten_leaves(v)]
    return [sample]


class _SourceProvenance:
    """Mixin resolving sample indices back to their source files.

    Snapshots only picklable primitives from the source dataset(s) — never the
    live mmap-holding dataset — so a collate stays lightweight and picklable
    across `num_workers > 0`.
    """

    def _init_sources(self, source: Any | None) -> None:
        """Snapshot `files`/`_file_offsets`/`new_axis` of 0, 1, or many sources."""
        if source is None:
            sources: list[Any] = []
        elif isinstance(source, list | tuple):
            sources = list(source)
        else:
            sources = [source]
        self._snaps: list[tuple[list[Path], list[int], bool]] = [
            (list(s.files), list(s._file_offsets), bool(getattr(s, "new_axis", False)))
            for s in sources
        ]

    @property
    def files(self) -> list[Path] | None:
        """Snapshotted file list of the first source (back-compat accessor)."""
        return self._snaps[0][0] if self._snaps else None

    @property
    def file_offsets(self) -> list[int] | None:
        """Snapshotted file offsets of the first source (back-compat accessor)."""
        return self._snaps[0][1] if self._snaps else None

    @property
    def new_axis(self) -> bool:
        """`new_axis` flag of the first source (back-compat accessor)."""
        return self._snaps[0][2] if self._snaps else False

    def _file_of(self, index: int, source: int = 0) -> Path:
        """Resolve a global sample index to the `Path` of the given source."""
        assert self._snaps, "no source snapshot available"
        files, offsets, new_axis = self._snaps[source]
        if new_axis:
            return files[index]
        for file_idx, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            if lo <= index < hi:
                return files[file_idx]
        raise IndexError(f"Index {index} out of range for source files")

    def _provenance(self, anchors: list[int]) -> dict[str, Any]:
        """Build the `key`/`files`/`indices` dict for one index per batch entry.

        Args:
            anchors: One representative sample index per batch entry.
        """
        per_source = [
            [self._file_of(i, s) for i in anchors] for s in range(len(self._snaps))
        ]
        # Single source -> flat list (back-compat); many -> list per source.
        files = per_source[0] if len(per_source) == 1 else per_source
        first = per_source[0][0] if per_source[0] else None
        key = self.key_fn(first) if (self.key_fn and first is not None) else None
        return {"key": key, "files": files}


class SequenceCollate(_SourceProvenance):
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
            source: Optional `FileDataset`, or a sequence of them (e.g. a
                `ZipDataset`'s `.datasets`), whose `.files`/`_file_offsets` are
                snapshotted (never the live dataset) so batches from
                `with_indices` carry `key`/`files`/`indices`. With a single
                source `files` is a flat list; with several it is one list per
                source, positionally aligned with the sources.
            key_fn: Optional `Path -> str` labelling a batch's key from its first
                file. Must be picklable for `num_workers>0`.
        """
        self.transform = transform
        self.key_fn = key_fn
        self._init_sources(source)

    def _apply(self, sample: Any, params: Any) -> Any:
        """Apply `transform(leaf, params)` to every tensor leaf of `sample`."""
        if isinstance(sample, dict):
            return {k: self._apply(v, params) for k, v in sample.items()}
        if isinstance(sample, tuple | list):
            return type(sample)(self._apply(v, params) for v in sample)
        if isinstance(sample, torch.Tensor):
            return self.transform.transform(sample, params)
        return sample

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

        if indices is not None and self._snaps:
            provenance = {**self._provenance(indices), "indices": list(indices)}
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

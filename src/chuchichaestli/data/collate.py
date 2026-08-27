# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collation utilities for sequence/loop batches."""

from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping, Sequence
import torch
from torch.utils.data import default_collate
from torchvision.transforms.v2 import Transform
from chuchichaestli.data.base import IndexedSample
from chuchichaestli.utils import map_nested


__all__ = [
    "SequenceCollate",
    "sequence_collate",
    "SlidingWindowCollate",
    "sliding_window_collate",
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
        """Snapshot `files`/`_file_offsets`/`sample_axis` of 0, 1, or many sources."""
        if source is None:
            sources: list[Any] = []
        elif isinstance(source, list | tuple):
            sources = list(source)
        else:
            sources = [source]
        snaps: list[tuple[list[Path], list[int], int | None]] = []
        for s in sources:
            files = getattr(s, "files", None)
            if not files:
                continue
            offsets = getattr(s, "_file_offsets", None)
            snaps.append(
                (list(files), list(offsets or []), getattr(s, "sample_axis", 0))
            )
        if snaps and len(snaps) != len(sources):
            raise ValueError(
                f"{len(sources) - len(snaps)} of {len(sources)} sources report "
                "no files; provenance is resolved per source position, so pass "
                "either all sources with files or none"
            )
        self._snaps = snaps

    @property
    def files(self) -> list[Path] | None:
        """Snapshotted file list of the first source (back-compat accessor)."""
        return self._snaps[0][0] if self._snaps else None

    @property
    def file_offsets(self) -> list[int] | None:
        """Snapshotted file offsets of the first source (back-compat accessor)."""
        return self._snaps[0][1] if self._snaps else None

    @property
    def sample_axis(self) -> int | None:
        """`sample_axis` of the first source (back-compat accessor)."""
        return self._snaps[0][2] if self._snaps else 0

    def _file_of(self, index: int, source: int = 0) -> Path:
        """Resolve a global sample index to the `Path` of the given source."""
        assert self._snaps, "no source snapshot available"
        files, offsets, sample_axis = self._snaps[source]
        if sample_axis is None:
            # One sample per file, so the index *is* the file index.
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


class SlidingWindowCollate(_SourceProvenance):
    """Pickleable `collate_fn` for `SlidingWindowBatchSampler` batches.

    The sampler emits `batch_size` windows flattened into a single index list,
    so `default_collate` yields tensors of shape `(batch_size * span, ...)` with
    `span = window_size + horizon`. This collate restores the window axis,
    giving `(batch_size, span, ...)`, then splits it into the input window and
    the trailing forecast target. Output follows the sample structure.
    """

    def __init__(
        self,
        window_size: int,
        horizon: int = 0,
        target_suffix: str = "_target",
        rename: dict[str, str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor]
        | Mapping[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        source: Any | None = None,
        key_fn: Callable[[Path], str] | None = None,
    ):
        """Constructor.

        Args:
            window_size: Number of leading samples per window forming the input.
            horizon: Number of trailing samples per window used as the target;
                `0` disables the split.
            target_suffix: Suffix appended to a dict key to name its target.
            rename: Explicit `input key -> target key` overrides, taking
                precedence over `target_suffix`.
            transform: Optional transform applied after reshaping and
                splitting. Use it to compress or normalize whole windows (e.g.
                a basis projection). Either a single callable, applied to every
                tensor leaf, or a `key -> callable` mapping applied only to
                those entries (of a `ZipDataset`).
            source: Optional `FileDataset`, or a sequence of them, whose
                `.files`/`_file_offsets` are snapshotted so batches from
                `with_indices` carry `key`/`files`/`indices`. `files` holds one
                entry per window; `indices` holds that window's sample indices.
            key_fn: Optional `Path -> str` labelling a batch's key from its first
                window's file. Must be picklable for `num_workers>0`.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        self.window_size = window_size
        self.horizon = horizon
        self.target_suffix = target_suffix
        self.rename = dict(rename) if rename else {}
        self.transform = transform
        self.key_fn = key_fn
        self._init_sources(source)

    @classmethod
    def from_sampler(cls, sampler: Any, **kwargs: Any) -> "SlidingWindowCollate":
        """Build a collate matching a sampler's window geometry.

        Args:
            sampler: A `SlidingWindowBatchSampler`.
            kwargs: Passed through to the constructor.
        """
        return cls(window_size=sampler.window_size, horizon=sampler.horizon, **kwargs)

    @property
    def span(self) -> int:
        """Number of samples in one window, including the horizon."""
        return self.window_size + self.horizon

    def _target_key(self, key: Any) -> Any:
        """Name the target entry belonging to input `key`."""
        if key in self.rename:
            return self.rename[key]
        return f"{key}{self.target_suffix}" if isinstance(key, str) else key

    def _finish(self, batch: Any) -> Any:
        """Apply the optional transform, per leaf or per key."""
        if self.transform is None:
            return batch
        if callable(self.transform):
            return map_nested(batch, self.transform)
        if not isinstance(batch, dict):
            raise TypeError(
                "a per-key `transform` needs dict samples, but this batch is a "
                f"{type(batch).__name__}; pass a single callable instead"
            )
        unknown = set(self.transform) - set(batch)
        if unknown:
            raise KeyError(
                f"`transform` names {sorted(unknown)}, which the batch does not "
                f"hold; available keys are {sorted(batch)}"
            )
        return {
            key: map_nested(value, self.transform[key])
            if key in self.transform
            else value
            for key, value in batch.items()
        }

    def __call__(self, samples: Sequence[Any]) -> Any:
        """Collate one flattened multi-window batch."""
        indices: list[int] | None = None
        if samples and isinstance(samples[0], IndexedSample):
            indices = [s.index for s in samples]
            samples = [s.sample for s in samples]

        n = len(samples)
        span = self.span
        if n % span:
            raise ValueError(
                f"Batch of {n} samples is not divisible by window span {span}; "
                "the sampler and collate must share window_size and horizon."
            )
        batch = default_collate(list(samples))
        batch = map_nested(batch, lambda t: t.reshape(n // span, span, *t.shape[1:]))

        provenance: dict[str, Any] = {}
        if indices is not None and self._snaps:
            # One window per batch entry
            per_window = [indices[i : i + span] for i in range(0, n, span)]
            provenance = {
                **self._provenance([w[0] for w in per_window]),
                "indices": per_window,
            }

        if self.horizon == 0:
            return self._merge(self._finish(batch), provenance)

        window = self._finish(map_nested(batch, lambda t: t[:, : self.window_size]))
        target = self._finish(map_nested(batch, lambda t: t[:, self.window_size :]))
        if isinstance(batch, dict):
            merged = dict(window)
            for key, value in target.items():
                target_key = self._target_key(key)
                if target_key in merged:
                    # Non-string keys (e.g. `ZipDataset(zip_as="dict")`) get no
                    # automatic suffix, so the target would replace its window.
                    raise ValueError(
                        f"the target for key {key!r} is named {target_key!r}, "
                        "which already holds the input window; pass "
                        f"`rename={{{key!r}: <target key>}}` to name it apart"
                    )
                merged[target_key] = value
            return self._merge(merged, provenance)
        if provenance:
            # Anonymous tensor samples are named "data"
            return {
                "data": window,
                self._target_key("data"): target,
                **provenance,
            }
        return window, target

    @staticmethod
    def _merge(batch: Any, provenance: dict[str, Any]) -> Any:
        """Attach provenance, wrapping non-dict batches like `SequenceCollate`."""
        if not provenance:
            return batch
        if isinstance(batch, dict):
            return {**batch, **provenance}
        return {"data": batch, **provenance}


def sequence_collate(
    transform: Transform | None = None,
    source: Any | None = None,
    key_fn: Callable[[Path], str] | None = None,
) -> SequenceCollate:
    """Return a `SequenceCollate` instance (see `SequenceCollate`)."""
    return SequenceCollate(transform=transform, source=source, key_fn=key_fn)


def sliding_window_collate(
    window_size: int,
    horizon: int = 0,
    target_suffix: str = "_target",
    rename: dict[str, str] | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor]
    | Mapping[str, Callable[[torch.Tensor], torch.Tensor]]
    | None = None,
    source: Any | None = None,
    key_fn: Callable[[Path], str] | None = None,
) -> SlidingWindowCollate:
    """Return a `SlidingWindowCollate` instance (see `SlidingWindowCollate`)."""
    return SlidingWindowCollate(
        window_size=window_size,
        horizon=horizon,
        target_suffix=target_suffix,
        rename=rename,
        transform=transform,
        source=source,
        key_fn=key_fn,
    )

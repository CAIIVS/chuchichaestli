# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the batching module."""

import re
import pytest
from pathlib import Path
from types import SimpleNamespace
from chuchichaestli.data.batching import (
    HierarchicalBatchSampler,
    HierarchicalFileBatchSampler,
)


def make_file_dataset(file_specs):
    """Build a minimal mock FileDataset.

    Args:
        file_specs: list of (path_str, n_samples) pairs.
    """
    files = [Path(p) for p, _ in file_specs]
    offsets = [0]
    for _, n in file_specs:
        offsets.append(offsets[-1] + n)
    return SimpleNamespace(files=files, _file_offsets=offsets)


class TestBatchIndices:
    """Tests for the static HierarchicalBatchSampler.batch_indices helper."""

    def test_one_arg_yields_unit_batches(self):
        """(end,) → unit batches from 0 to end."""
        batches = HierarchicalBatchSampler.batch_indices(5)
        assert batches == [[0], [1], [2], [3], [4]]

    def test_two_args_yields_sized_batches(self):
        """(end, size) → batches of given size starting at 0."""
        batches = HierarchicalBatchSampler.batch_indices(6, 2)
        assert batches == [[0, 1], [2, 3], [4, 5]]

    def test_three_args_with_offset(self):
        """(start, end, size) → batches from start."""
        batches = HierarchicalBatchSampler.batch_indices(10, 16, 3)
        assert batches == [[10, 11, 12], [13, 14, 15]]

    def test_partial_last_batch_kept_by_default(self):
        """Remainder batch is included when drop_last=False."""
        batches = HierarchicalBatchSampler.batch_indices(7, 3)
        assert batches[-1] == [6]
        assert len(batches) == 3

    def test_drop_last_removes_partial_batch(self):
        """drop_last=True discards the final short batch."""
        batches = HierarchicalBatchSampler.batch_indices(7, 3, drop_last=True)
        assert batches == [[0, 1, 2], [3, 4, 5]]

    def test_drop_last_no_effect_when_evenly_divisible(self):
        """drop_last has no effect when the range divides evenly."""
        keep = HierarchicalBatchSampler.batch_indices(6, 3)
        drop = HierarchicalBatchSampler.batch_indices(6, 3, drop_last=True)
        assert keep == drop

    def test_zero_end_returns_empty(self):
        """An end of 0 produces no batches."""
        assert HierarchicalBatchSampler.batch_indices(0) == []

    def test_too_many_args_raises_type_error(self):
        """More than 3 positional args raises TypeError."""
        with pytest.raises(TypeError):
            HierarchicalBatchSampler.batch_indices(0, 10, 3, 99)


class TestHierarchicalBatchSampler:
    """Tests for HierarchicalBatchSampler."""

    def test_len(self):
        """__len__ returns the number of batches."""
        sampler = HierarchicalBatchSampler([[0, 1], [2, 3], [4]])
        assert len(sampler) == 3

    def test_len_empty(self):
        """Empty batch list has length 0."""
        assert len(HierarchicalBatchSampler([])) == 0

    def test_iter_preserves_order(self):
        """Without shuffle, batches are yielded in insertion order."""
        batches = [[0, 1], [2, 3], [4, 5]]
        sampler = HierarchicalBatchSampler(batches, shuffle=False)
        assert list(sampler) == batches

    def test_iter_empty(self):
        """Empty sampler yields nothing."""
        assert list(HierarchicalBatchSampler([])) == []

    def test_iter_shuffle_contains_same_batches(self):
        """With shuffle, all original batches appear exactly once."""
        batches = [[i, i + 1] for i in range(0, 20, 2)]
        sampler = HierarchicalBatchSampler(batches, shuffle=True)
        assert sorted(list(sampler)) == sorted(batches)

    def test_iter_shuffle_does_not_mutate_original(self):
        """Shuffling operates on a copy and leaves _batches unchanged."""
        batches = [[0, 1], [2, 3], [4, 5]]
        sampler = HierarchicalBatchSampler(batches, shuffle=True)
        original = list(batches)
        list(sampler)
        assert sampler._batches == original

    def test_multiple_iterations(self):
        """Sampler can be iterated more than once."""
        batches = [[0, 1], [2, 3]]
        sampler = HierarchicalBatchSampler(batches)
        assert list(sampler) == list(sampler)


class TestHierarchicalFileBatchSampler:
    """Tests for HierarchicalFileBatchSampler."""

    def test_requires_files_attribute(self):
        """Datasets without a 'files' attribute raise TypeError."""
        bad_ds = SimpleNamespace(data=[1, 2, 3])
        with pytest.raises(TypeError, match="files"):
            HierarchicalFileBatchSampler(bad_ds, key_fn=lambda p: "k")

    def test_len_matches_batch_count(self):
        """__len__ equals the total number of built batches."""
        ds = make_file_dataset([("a.npy", 6), ("b.npy", 4)])
        sampler = HierarchicalFileBatchSampler(ds, key_fn=lambda p: p.stem, batch_size=2)
        # file a → 3 batches, file b → 2 batches
        assert len(sampler) == 5

    def test_iter_covers_all_indices(self):
        """All sample indices appear exactly once across batches."""
        ds = make_file_dataset([("a.npy", 6), ("b.npy", 4)])
        sampler = HierarchicalFileBatchSampler(ds, key_fn=lambda p: p.stem, batch_size=2)
        indices = sorted(i for batch in sampler for i in batch)
        assert indices == list(range(10))

    def test_batch_size_respected(self):
        """Every batch has exactly batch_size indices when evenly divisible."""
        ds = make_file_dataset([("a.npy", 9)])
        sampler = HierarchicalFileBatchSampler(ds, key_fn=lambda p: "g", batch_size=3)
        assert all(len(b) == 3 for b in sampler)

    def test_drop_last_removes_partial_batch(self):
        """drop_last=True drops the trailing short batch per group."""
        ds = make_file_dataset([("a.npy", 7)])
        sampler = HierarchicalFileBatchSampler(
            ds, key_fn=lambda p: "g", batch_size=3, drop_last=True
        )
        assert len(sampler) == 2
        indices = sorted(i for batch in sampler for i in batch)
        assert indices == [0, 1, 2, 3, 4, 5]

    def test_grouping_keeps_file_indices_separate(self):
        """Samples from files sharing a key are batched together."""
        ds = make_file_dataset([
            ("group_a_0.npy", 4),
            ("group_a_1.npy", 4),
            ("group_b_0.npy", 4),
        ])

        def key_fn(p):
            return "_".join(p.stem.split("_")[:2])  # "group_a" / "group_b"

        sampler = HierarchicalFileBatchSampler(ds, key_fn=key_fn, batch_size=4)
        batches = list(sampler)
        assert len(batches) == 3
        group_a = set(range(8))
        group_b = set(range(8, 12))
        for batch in batches:
            assert set(batch) <= group_a or set(batch) <= group_b

    def test_shuffle_true_all_indices_present(self):
        """With shuffle=True, all indices still appear across batches."""
        ds = make_file_dataset([("a.npy", 6), ("b.npy", 6)])
        sampler = HierarchicalFileBatchSampler(
            ds, key_fn=lambda p: p.stem, batch_size=2, shuffle=True
        )
        indices = sorted(i for batch in sampler for i in batch)
        assert indices == list(range(12))

    def test_single_file_single_batch(self):
        """A single file whose samples fit in one batch yields one batch."""
        ds = make_file_dataset([("only.npy", 4)])
        sampler = HierarchicalFileBatchSampler(ds, key_fn=lambda p: "g", batch_size=4)
        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [0, 1, 2, 3]


class TestHierarchicalFileBatchSamplerFromFilename:
    """Tests for the from_filename factory method."""

    def test_returns_correct_type(self):
        """from_filename returns a HierarchicalFileBatchSampler."""
        ds = make_file_dataset([("run_001.npy", 2)])
        sampler = HierarchicalFileBatchSampler.from_filename(ds, pattern=r"(run_\d+)")
        assert isinstance(sampler, HierarchicalFileBatchSampler)

    def test_groups_by_captured_pattern(self):
        """Files matched by the same capture group are batched together."""
        ds = make_file_dataset([
            ("run_001_frame_0.npy", 4),
            ("run_001_frame_1.npy", 4),
            ("run_002_frame_0.npy", 4),
        ])
        sampler = HierarchicalFileBatchSampler.from_filename(
            ds, pattern=r"(run_\d+)", batch_size=4
        )
        # run_001: 8 samples → 2 batches; run_002: 4 → 1 batch
        assert len(sampler) == 3

    def test_fallback_for_unmatched_files(self):
        """Files not matching the pattern are assigned the fallback key."""
        ds = make_file_dataset([("nomatch.npy", 4), ("run_001.npy", 4)])
        sampler = HierarchicalFileBatchSampler.from_filename(
            ds, pattern=r"(run_\d+)", fallback="__other__", batch_size=4
        )
        # two distinct keys → 2 batches
        assert len(sampler) == 2

    def test_all_files_unmatched_uses_fallback_group(self):
        """When no file matches, all fall into a single fallback group."""
        ds = make_file_dataset([("foo.npy", 3), ("bar.npy", 3)])
        sampler = HierarchicalFileBatchSampler.from_filename(
            ds, pattern=r"(run_\d+)", fallback="__other__", batch_size=3
        )
        assert len(sampler) == 2
        indices = sorted(i for batch in sampler for i in batch)
        assert indices == list(range(6))

    def test_batch_size_forwarded(self):
        """batch_size is forwarded correctly."""
        ds = make_file_dataset([("run_001.npy", 6)])
        sampler = HierarchicalFileBatchSampler.from_filename(
            ds, pattern=r"(run_\d+)", batch_size=3
        )
        for batch in sampler:
            assert len(batch) == 3

    def test_drop_last_forwarded(self):
        """drop_last=True is forwarded correctly."""
        ds = make_file_dataset([("run_001.npy", 7)])
        sampler = HierarchicalFileBatchSampler.from_filename(
            ds, pattern=r"(run_\d+)", batch_size=3, drop_last=True
        )
        assert len(sampler) == 2


class TestHierarchicalFileBatchSamplerFromDirectory:
    """Tests for the from_directory factory method."""

    def test_returns_correct_type(self):
        """from_directory returns a HierarchicalFileBatchSampler."""
        ds = make_file_dataset([("/data/cls/img.npy", 4)])
        sampler = HierarchicalFileBatchSampler.from_directory(ds)
        assert isinstance(sampler, HierarchicalFileBatchSampler)

    def test_groups_by_immediate_parent(self):
        """depth=1 groups files by their immediate parent directory."""
        ds = make_file_dataset([
            ("/data/classA/img0.npy", 4),
            ("/data/classA/img1.npy", 4),
            ("/data/classB/img0.npy", 4),
        ])
        sampler = HierarchicalFileBatchSampler.from_directory(ds, depth=1, batch_size=4)
        # classA: 8 samples → 2 batches; classB: 4 → 1
        assert len(sampler) == 3

    def test_depth_two_groups_by_grandparent(self):
        """depth=2 groups by the grandparent directory."""
        ds = make_file_dataset([
            ("/root/split/classA/img0.npy", 4),
            ("/root/split/classA/img1.npy", 4),
            ("/root/other/classA/img0.npy", 4),
        ])
        sampler = HierarchicalFileBatchSampler.from_directory(ds, depth=2, batch_size=4)
        # "split" vs "other" → 2 distinct keys → 3 batches
        assert len(sampler) == 3

    def test_all_indices_covered(self):
        """All sample indices appear across all batches."""
        ds = make_file_dataset([
            ("/data/A/f0.npy", 5),
            ("/data/B/f0.npy", 5),
        ])
        sampler = HierarchicalFileBatchSampler.from_directory(ds, depth=1, batch_size=2)
        indices = sorted(i for batch in sampler for i in batch)
        assert indices == list(range(10))

    def test_drop_last_forwarded(self):
        """drop_last=True is forwarded correctly."""
        ds = make_file_dataset([("/data/A/f0.npy", 7)])
        sampler = HierarchicalFileBatchSampler.from_directory(
            ds, depth=1, batch_size=3, drop_last=True
        )
        assert len(sampler) == 2

    def test_batch_size_forwarded(self):
        """batch_size is forwarded correctly."""
        ds = make_file_dataset([("/data/A/f0.npy", 6)])
        sampler = HierarchicalFileBatchSampler.from_directory(ds, depth=1, batch_size=2)
        for batch in sampler:
            assert len(batch) == 2


# A per-file numeric ordering by the trailing "_<N>" integer (arts4ska style).
_ORDER_RX = re.compile(r"_(\d+)\.npy$")


def _order_n(path: Path) -> int:
    return int(_ORDER_RX.search(path.name).group(1))


class TestOrderFn:
    """Tests for the order_fn intra-group ordering knob."""

    def test_order_fn_sorts_group_numerically(self):
        """order_fn reorders a group's samples by the numeric key (2,3,10)."""
        # Files listed in lexical glob order: _10 precedes _2, _3.
        ds = make_file_dataset([
            ("xfrac_z9.940_10.npy", 1),
            ("xfrac_z9.940_2.npy", 1),
            ("xfrac_z9.940_3.npy", 1),
        ])
        sampler = HierarchicalFileBatchSampler(
            ds, key_fn=lambda p: "z9.940", batch_size=None, order_fn=_order_n
        )
        batches = list(sampler)
        assert len(batches) == 1
        # file idx 0=_10, 1=_2, 2=_3 -> sorted by N: _2(1), _3(2), _10(0)
        assert batches[0] == [1, 2, 0]

    def test_order_fn_none_matches_legacy_output(self):
        """order_fn=None (default) reproduces the pre-existing partitioning."""
        ds = make_file_dataset([("a.npy", 6), ("b.npy", 4)])
        sampler = HierarchicalFileBatchSampler(
            ds, key_fn=lambda p: p.stem, batch_size=2
        )
        assert list(sampler) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


class TestWholeGroupBatching:
    """Tests for the batch_size=None whole-group sentinel."""

    def test_merges_group_into_single_batch(self):
        """batch_size=None yields one batch per group, merging all indices."""
        ds = make_file_dataset([
            ("run_a_1.npy", 1),
            ("run_a_2.npy", 1),
            ("run_a_3.npy", 1),
            ("run_b_1.npy", 1),
            ("run_b_2.npy", 1),
        ])
        key_fn = lambda p: p.name.split("_")[1]  # "a" / "b"  # noqa: E731
        sampler = HierarchicalFileBatchSampler(ds, key_fn=key_fn, batch_size=None)
        batches = list(sampler)
        # Different loops -> batches of different lengths.
        assert sorted(len(b) for b in batches) == [2, 3]
        assert sorted(i for b in batches for i in b) == list(range(5))

    def test_merges_multi_sample_files(self):
        """Whole-group mode merges every sample index across a group's files."""
        ds = make_file_dataset([("a.npy", 3), ("b.npy", 2)])
        sampler = HierarchicalFileBatchSampler(
            ds, key_fn=lambda p: "g", batch_size=None
        )
        assert list(sampler) == [[0, 1, 2, 3, 4]]

    def test_drop_last_is_noop_in_whole_group_mode(self):
        """drop_last has no effect when batch_size is None."""
        specs = [("g_1.npy", 1), ("g_2.npy", 1), ("g_3.npy", 1)]
        keep = HierarchicalFileBatchSampler(
            make_file_dataset(specs), key_fn=lambda p: "g",
            batch_size=None, order_fn=_order_n, drop_last=False,
        )
        drop = HierarchicalFileBatchSampler(
            make_file_dataset(specs), key_fn=lambda p: "g",
            batch_size=None, order_fn=_order_n, drop_last=True,
        )
        assert list(keep) == list(drop) == [[0, 1, 2]]

    def test_shuffle_preserves_intra_batch_order(self):
        """shuffle reorders batches but each batch stays order_fn-monotonic."""
        ds = make_file_dataset([
            ("s1_10.npy", 1),
            ("s1_2.npy", 1),
            ("s1_3.npy", 1),
            ("s2_2.npy", 1),
            ("s2_1.npy", 1),
        ])
        key_fn = lambda p: p.name.split("_")[0]  # "s1" / "s2"  # noqa: E731
        sampler = HierarchicalFileBatchSampler(
            ds, key_fn=key_fn, batch_size=None, order_fn=_order_n, shuffle=True
        )
        for _ in range(5):
            for batch in sampler:
                ns = [_order_n(ds.files[i]) for i in batch]
                assert ns == sorted(ns)


class TestHierarchicalFileBatchSamplerFromSequences:
    """Tests for the from_sequences factory method."""

    def test_one_batch_per_sequence_key(self):
        """Each z-label group becomes one N-ascending batch."""
        ds = make_file_dataset([
            ("xfrac_z9.940_2.npy", 1),
            ("xfrac_z9.940_10.npy", 1),
            ("xfrac_z9.940_3.npy", 1),
            ("xfrac_z8.100_2.npy", 1),
            ("xfrac_z8.100_4.npy", 1),
        ])
        sampler = HierarchicalFileBatchSampler.from_sequences(
            ds, pattern=r"_z([0-9.]+)_(\d+)", group=1, order_group=2, order_cast=int
        )
        batches = list(sampler)
        assert len(batches) == 2
        # z9.940: idx 0=_2, 1=_10, 2=_3 -> ascending N: [0, 2, 1]
        assert batches[0] == [0, 2, 1]
        # z8.100: idx 3=_2, 4=_4 -> [3, 4]
        assert batches[1] == [3, 4]

    def test_warns_on_unmatched_files(self):
        """Unmatched files trigger a warning and form one fallback batch."""
        ds = make_file_dataset([
            ("xfrac_z9.940_2.npy", 1),
            ("README.npy", 1),
        ])
        with pytest.warns(UserWarning, match="did not match"):
            sampler = HierarchicalFileBatchSampler.from_sequences(
                ds, pattern=r"_z([0-9.]+)_(\d+)"
            )
        batches = list(sampler)
        # matched sequence + fallback sequence
        assert len(batches) == 2
        assert sorted(i for b in batches for i in b) == [0, 1]

    def test_returns_correct_type(self):
        """from_sequences returns a HierarchicalFileBatchSampler."""
        ds = make_file_dataset([("xfrac_z9.940_2.npy", 1)])
        sampler = HierarchicalFileBatchSampler.from_sequences(
            ds, pattern=r"_z([0-9.]+)_(\d+)"
        )
        assert isinstance(sampler, HierarchicalFileBatchSampler)

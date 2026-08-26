# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the collate module."""

import re
import pickle
import pytest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import default_collate
from torchvision.transforms.v2 import Transform
from chuchichaestli.data.base import IndexedSample, WithIndices, with_indices
from chuchichaestli.data.transforms import RandomCropND
from chuchichaestli.data.collate import (
    SequenceCollate,
    sequence_collate,
    SlidingWindowCollate,
    sliding_window_collate,
    _flatten_leaves,
)


class _AddOnce(Transform):
    """Test transform: records `make_params` calls; adds a shared value."""

    _transformed_types = (torch.Tensor,)

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def make_params(self, flat_inputs):
        self.calls.append(len(flat_inputs))
        return {"add": self.value}

    def transform(self, inpt, params):
        return inpt + params["add"]


def _z_key(path: Path) -> str:
    """Module-level (picklable) key_fn extracting the z-label."""
    return re.search(r"_z([0-9.]+)_", path.name).group(1)


class TestWithIndices:
    """Tests for the with_indices wrapper."""

    def test_wraps_samples_with_index(self):
        """__getitem__ returns IndexedSample(index, sample)."""

        class DS:
            def __len__(self):
                return 3

            def __getitem__(self, i):
                return {"v": i}

        w = with_indices(DS())
        assert isinstance(w, WithIndices)
        assert len(w) == 3
        sample = w[2]
        assert isinstance(sample, IndexedSample)
        assert sample.index == 2
        assert sample.sample == {"v": 2}


class TestFlattenLeaves:
    """Tests for the _flatten_leaves helper."""

    def test_dict(self):
        """All dict values are flattened in order."""
        a, b = torch.zeros(2, 3), torch.ones(1)
        leaves = _flatten_leaves({"a": a, "b": b})
        assert leaves[0] is a and leaves[1] is b and len(leaves) == 2

    def test_nested(self):
        """Recurses through tuples/lists."""
        t, u = torch.zeros(4), torch.ones(1)
        leaves = _flatten_leaves({"a": (t, u)})
        assert leaves[0] is t and leaves[1] is u and len(leaves) == 2

    def test_tensor_passthrough(self):
        """A bare tensor flattens to a single-element list."""
        t = torch.zeros(4)
        leaves = _flatten_leaves(t)
        assert leaves == [t] and leaves[0] is t


class TestSequenceCollateNoTransform:
    """Tests for the identity (transform=None) path."""

    def test_reduces_to_default_collate(self):
        """transform=None behaves exactly like default_collate."""
        samples = [
            {"x": torch.full((2, 2), float(i)), "y": torch.tensor([float(i)])}
            for i in range(4)
        ]
        out = SequenceCollate()(samples)
        ref = default_collate(samples)
        assert out.keys() == ref.keys()
        assert torch.equal(out["x"], ref["x"])
        assert torch.equal(out["y"], ref["y"])
        assert out["x"].shape == (4, 2, 2)

    def test_variable_length_batches(self):
        """Different batches stack to different leading T."""
        short = [{"x": torch.zeros(3)} for _ in range(2)]
        long = [{"x": torch.zeros(3)} for _ in range(5)]
        coll = SequenceCollate()
        assert coll(short)["x"].shape == (2, 3)
        assert coll(long)["x"].shape == (5, 3)


class TestSequenceCollateSharedTransform:
    """Tests for the v2-Transform path (shared per-batch params)."""

    def test_make_params_called_once_and_shared(self):
        """make_params runs once; the same params reach every item + field."""
        t = _AddOnce(5.0)
        samples = [{"a": torch.zeros(2), "b": torch.zeros(2)} for _ in range(3)]
        out = SequenceCollate(transform=t)(samples)
        assert len(t.calls) == 1  # sampled once per batch, not per item
        assert t.calls[0] == 2  # both leaves of the first sample
        # every field of every step got the same +5
        assert torch.equal(out["a"], torch.full((3, 2), 5.0))
        assert torch.equal(out["b"], torch.full((3, 2), 5.0))

    def test_shared_crop_keeps_all_fields_aligned(self):
        """RandomCropND crops *both* paired fields with the same box.

        Guards against torchvision's `forward` heuristic, which would transform
        only the first pure tensor of a sample.
        """
        samples = [
            {"xfrac": torch.arange(8).float(), "ionrates": torch.arange(8).float()}
            for _ in range(2)
        ]
        out = SequenceCollate(transform=RandomCropND((4,)))(samples)
        assert out["xfrac"].shape == (2, 4)
        assert out["ionrates"].shape == (2, 4)  # not passed through uncropped
        assert torch.equal(out["xfrac"], out["ionrates"])

    def test_non_tensor_leaves_pass_through(self):
        """Non-tensor leaves are left untouched by the transform."""
        t = _AddOnce(1.0)
        samples = [{"x": torch.zeros(2), "label": i} for i in range(2)]
        out = SequenceCollate(transform=t)(samples)
        assert torch.equal(out["x"], torch.ones(2, 2))
        assert out["label"].tolist() == [0, 1]


class TestSequenceCollateProvenance:
    """Tests for provenance attachment via with_indices + source."""

    def _source(self):
        files = [Path(f"xfrac_z9.940_{n}.npy") for n in (2, 3, 10)]
        return SimpleNamespace(files=files, _file_offsets=[0, 1, 2, 3], sample_axis=None)

    def test_attaches_key_files_indices(self):
        """A provenance batch carries key/files/indices matching the indices."""
        coll = SequenceCollate(source=self._source(), key_fn=_z_key)
        # Emulate a with_indices batch for global indices [0, 2, 1].
        samples = [
            IndexedSample(i, {"xfrac": torch.zeros(2), "ionrates": torch.ones(2)})
            for i in (0, 2, 1)
        ]
        out = coll(samples)
        assert out["indices"] == [0, 2, 1]
        assert [p.name for p in out["files"]] == [
            "xfrac_z9.940_2.npy",
            "xfrac_z9.940_10.npy",
            "xfrac_z9.940_3.npy",
        ]
        assert out["key"] == "9.940"
        # data is still collated
        assert out["xfrac"].shape == (3, 2)
        assert out["ionrates"].shape == (3, 2)

    def test_multi_source_attaches_per_field_files(self):
        """A sequence of sources yields one path list per source."""
        xfrac = SimpleNamespace(
            files=[Path(f"xfrac_z9.940_{n}.npy") for n in (2, 3, 10)],
            _file_offsets=[0, 1, 2, 3],
            sample_axis=None,
        )
        ionrates = SimpleNamespace(
            files=[Path(f"IonRates_z9.940_{n}.npy") for n in (2, 3, 10)],
            _file_offsets=[0, 1, 2, 3],
            sample_axis=None,
        )
        coll = SequenceCollate(source=[xfrac, ionrates], key_fn=_z_key)
        samples = [
            IndexedSample(i, {"xfrac": torch.zeros(2), "ionrates": torch.ones(2)})
            for i in (0, 2, 1)
        ]
        out = coll(samples)
        xfrac_files, ionrates_files = out["files"]
        assert [p.name for p in xfrac_files] == [
            "xfrac_z9.940_2.npy",
            "xfrac_z9.940_10.npy",
            "xfrac_z9.940_3.npy",
        ]
        assert [p.name for p in ionrates_files] == [
            "IonRates_z9.940_2.npy",
            "IonRates_z9.940_10.npy",
            "IonRates_z9.940_3.npy",
        ]
        # key is still derived from the first source's first file
        assert out["key"] == "9.940"

    def test_single_element_sequence_stays_flat(self):
        """A one-element source sequence keeps the flat (back-compat) shape."""
        coll = SequenceCollate(source=[self._source()])
        samples = [IndexedSample(i, {"x": torch.zeros(2)}) for i in (0, 1)]
        out = coll(samples)
        assert [p.name for p in out["files"]] == [
            "xfrac_z9.940_2.npy",
            "xfrac_z9.940_3.npy",
        ]

    def test_no_source_skips_provenance(self):
        """Without source, IndexedSample data is collated but no provenance."""
        coll = SequenceCollate()
        samples = [IndexedSample(i, {"x": torch.zeros(2)}) for i in (0, 1)]
        out = coll(samples)
        assert "indices" not in out and "files" not in out
        assert out["x"].shape == (2, 2)

    def test_map_index_round_trip_along_axis_0(self):
        """Index -> file resolves correctly when samples run along axis 0."""
        files = [Path("a.npy"), Path("b.npy")]
        source = SimpleNamespace(files=files, _file_offsets=[0, 3, 5], sample_axis=0)
        coll = SequenceCollate(source=source)
        # indices 0,1,2 -> a.npy ; 3,4 -> b.npy
        assert coll._file_of(0).name == "a.npy"
        assert coll._file_of(2).name == "a.npy"
        assert coll._file_of(3).name == "b.npy"
        assert coll._file_of(4).name == "b.npy"


class TestSequenceCollatePicklable:
    """Tests guarding DataLoader(num_workers>0) usage."""

    def test_instance_survives_pickle(self):
        """A plain SequenceCollate round-trips through pickle and still works."""
        coll = SequenceCollate()
        restored = pickle.loads(pickle.dumps(coll))
        out = restored([{"x": torch.zeros(2)} for _ in range(2)])
        assert out["x"].shape == (2, 2)

    def test_provenance_snapshot_picklable(self):
        """Provenance snapshot (files/offsets) is picklable without the dataset."""
        files = [Path("xfrac_z9.940_2.npy")]
        source = SimpleNamespace(files=files, _file_offsets=[0, 1], sample_axis=None)
        coll = SequenceCollate(source=source)
        restored = pickle.loads(pickle.dumps(coll))
        assert restored.files == files
        assert restored.sample_axis is None

    def test_sequence_collate_convenience_returns_instance(self):
        """sequence_collate(...) returns a SequenceCollate."""
        assert isinstance(sequence_collate(), SequenceCollate)


def flat_windows(n_windows, span, make):
    """Build a flattened multi-window batch of samples via `make(index)`."""
    return [make(w * 100 + i) for w in range(n_windows) for i in range(span)]


class TestSlidingWindowCollate:
    """Tests for SlidingWindowCollate."""

    def test_restores_window_axis_and_splits(self):
        """Tensor samples split into (window, target) with a window axis."""
        coll = SlidingWindowCollate(window_size=3, horizon=1)
        samples = flat_windows(2, 4, lambda i: torch.tensor([float(i)]))
        window, target = coll(samples)
        assert window.shape == (2, 3, 1)
        assert target.shape == (2, 1, 1)

    def test_split_takes_leading_window_and_trailing_horizon(self):
        """The first window_size samples are input, the rest target."""
        coll = SlidingWindowCollate(window_size=3, horizon=1)
        samples = [torch.tensor([float(i)]) for i in range(4)]
        window, target = coll(samples)
        assert window[0, :, 0].tolist() == [0.0, 1.0, 2.0]
        assert target[0, :, 0].tolist() == [3.0]

    def test_horizon_zero_returns_single_batch(self):
        """horizon=0 inserts the window axis without splitting."""
        coll = SlidingWindowCollate(window_size=4)
        batch = coll([torch.tensor([float(i)]) for i in range(8)])
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (2, 4, 1)

    def test_dict_samples_merge_with_suffix(self):
        """Dict samples yield one merged dict keyed by suffix."""
        coll = SlidingWindowCollate(window_size=2, horizon=1)
        samples = [
            {"vis": torch.tensor([float(i)]), "uvw": torch.tensor([float(-i)])}
            for i in range(3)
        ]
        out = coll(samples)
        assert set(out) == {"vis", "uvw", "vis_target", "uvw_target"}
        assert out["vis"].shape == (1, 2, 1)
        assert out["vis_target"].shape == (1, 1, 1)

    def test_rename_overrides_suffix(self):
        """Rename replaces the suffix for the named keys only."""
        coll = SlidingWindowCollate(window_size=2, horizon=1, rename={"vis": "target"})
        samples = [
            {"vis": torch.tensor([float(i)]), "uvw": torch.tensor([float(i)])}
            for i in range(3)
        ]
        out = coll(samples)
        assert set(out) == {"vis", "uvw", "target", "uvw_target"}

    def test_transform_applied_to_every_leaf(self):
        """The optional transform runs on each tensor leaf of the output."""
        coll = SlidingWindowCollate(window_size=2, horizon=1, transform=lambda t: t * 0)
        window, target = coll([torch.tensor([float(i)]) for i in range(3)])
        assert torch.equal(window, torch.zeros_like(window))
        assert torch.equal(target, torch.zeros_like(target))

    def test_indexed_samples_are_unwrapped(self):
        """IndexedSample pairs are unwrapped before collation."""
        coll = SlidingWindowCollate(window_size=2, horizon=1)
        samples = [IndexedSample(i, torch.tensor([float(i)])) for i in range(3)]
        window, target = coll(samples)
        assert window.shape == (1, 2, 1)

    def test_batch_not_divisible_by_span_raises(self):
        """A batch that is not a whole number of windows is rejected."""
        coll = SlidingWindowCollate(window_size=3, horizon=1)
        with pytest.raises(ValueError, match="not divisible"):
            coll([torch.tensor([1.0])] * 5)

    def test_from_sampler_copies_geometry(self):
        """from_sampler picks up window_size and horizon."""
        sampler = SimpleNamespace(window_size=5, horizon=2)
        coll = SlidingWindowCollate.from_sampler(sampler)
        assert (coll.window_size, coll.horizon, coll.span) == (5, 2, 7)

    def test_is_picklable(self):
        """The collate survives pickling, so num_workers > 0 works."""
        coll = SlidingWindowCollate(window_size=2, horizon=1)
        restored = pickle.loads(pickle.dumps(coll))
        assert (restored.window_size, restored.horizon) == (2, 1)

    @pytest.mark.parametrize(
        "kwargs", [{"window_size": 0}, {"window_size": 2, "horizon": -1}]
    )
    def test_invalid_arguments_raise(self, kwargs):
        """Out-of-range geometry arguments raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowCollate(**kwargs)

    def test_convenience_returns_instance(self):
        """sliding_window_collate(...) returns a SlidingWindowCollate."""
        assert isinstance(sliding_window_collate(2), SlidingWindowCollate)


class TestSlidingWindowCollateProvenance:
    """Tests for window-level provenance via with_indices + source."""

    def _source(self, per_file=4):
        """Two files of `per_file` samples each."""
        files = [Path(f"xfrac_z9.940_{n}.npy") for n in (2, 3)]
        return SimpleNamespace(
            files=files,
            _file_offsets=[0, per_file, 2 * per_file],
            sample_axis=0,
        )

    def _indexed(self, indices):
        """Wrap sample indices as IndexedSample tensors."""
        return [IndexedSample(i, torch.tensor([float(i)])) for i in indices]

    def test_files_are_per_window(self):
        """Each window contributes one file entry, not one per sample."""
        coll = SlidingWindowCollate(
            window_size=2, horizon=1, source=self._source(), key_fn=_z_key
        )
        # two windows: [0,1,2] from file 0, [4,5,6] from file 1
        out = coll(self._indexed([0, 1, 2, 4, 5, 6]))
        assert [p.name for p in out["files"]] == [
            "xfrac_z9.940_2.npy",
            "xfrac_z9.940_3.npy",
        ]
        assert out["key"] == "9.940"

    def test_indices_are_grouped_by_window(self):
        """Indices is one list of sample indices per window."""
        coll = SlidingWindowCollate(window_size=2, horizon=1, source=self._source())
        out = coll(self._indexed([0, 1, 2, 4, 5, 6]))
        assert out["indices"] == [[0, 1, 2], [4, 5, 6]]

    def test_provenance_merges_into_dict_samples(self):
        """Dict samples keep their keys and gain provenance alongside."""
        coll = SlidingWindowCollate(window_size=2, horizon=1, source=self._source())
        samples = [
            IndexedSample(i, {"vis": torch.tensor([float(i)])}) for i in (0, 1, 2)
        ]
        out = coll(samples)
        assert set(out) == {"vis", "vis_target", "key", "files", "indices"}
        assert out["vis"].shape == (1, 2, 1)

    def test_anonymous_samples_named_data(self):
        """Tensor samples are wrapped as data/data_target when provenance is on."""
        coll = SlidingWindowCollate(window_size=2, horizon=1, source=self._source())
        out = coll(self._indexed([0, 1, 2]))
        assert set(out) == {"data", "data_target", "key", "files", "indices"}
        assert out["data"].shape == (1, 2, 1)
        assert out["data_target"].shape == (1, 1, 1)

    def test_horizon_zero_wraps_as_data(self):
        """With horizon=0 the batch is wrapped under 'data'."""
        coll = SlidingWindowCollate(window_size=3, source=self._source())
        out = coll(self._indexed([0, 1, 2]))
        assert set(out) == {"data", "key", "files", "indices"}
        assert out["data"].shape == (1, 3, 1)

    def test_without_source_returns_plain_batch(self):
        """Indexed samples without a source collate to a plain pair."""
        coll = SlidingWindowCollate(window_size=2, horizon=1)
        window, target = coll(self._indexed([0, 1, 2]))
        assert window.shape == (1, 2, 1)
        assert target.shape == (1, 1, 1)

    def test_survives_pickling_with_source(self):
        """A collate carrying a source snapshot stays picklable."""
        coll = SlidingWindowCollate(window_size=2, horizon=1, source=self._source())
        restored = pickle.loads(pickle.dumps(coll))
        assert restored.files == coll.files
        assert restored.file_offsets == coll.file_offsets


class TestSlidingWindowCollatePerKeyTransform:
    """Transforms targeted at individual entries of a heterogeneous sample."""

    @staticmethod
    def _samples(n=4):
        """`n` dict samples with two differently shaped entries."""
        return [{"a": torch.ones(3), "b": torch.ones(5)} for _ in range(n)]

    def test_single_callable_still_hits_every_leaf(self):
        """The original behaviour is unchanged."""
        collate = SlidingWindowCollate(window_size=4, transform=lambda t: t * 2)
        out = collate(self._samples())
        assert out["a"].unique().tolist() == [2.0]
        assert out["b"].unique().tolist() == [2.0]

    def test_mapping_applies_only_to_the_named_entry(self):
        """One projection rarely suits every entry of a zipped sample."""
        collate = SlidingWindowCollate(window_size=4, transform={"a": lambda t: t * 2})
        out = collate(self._samples())
        assert out["a"].unique().tolist() == [2.0]
        assert out["b"].unique().tolist() == [1.0]

    def test_mapping_covers_the_targets_too(self):
        """A key's transform applies to its input and its forecast target."""
        collate = SlidingWindowCollate(
            window_size=3, horizon=1, transform={"a": lambda t: t * 2}
        )
        out = collate(self._samples())
        assert out["a"].unique().tolist() == [2.0]
        assert out["a_target"].unique().tolist() == [2.0]
        assert out["b_target"].unique().tolist() == [1.0]

    def test_unknown_key_raises(self):
        """A typo must not silently transform nothing."""
        collate = SlidingWindowCollate(window_size=4, transform={"c": lambda t: t * 2})
        with pytest.raises(KeyError, match="does not"):
            collate(self._samples())

    def test_mapping_on_non_dict_samples_raises(self):
        """Bare tensor samples have no keys to address."""
        collate = SlidingWindowCollate(window_size=4, transform={"a": lambda t: t})
        with pytest.raises(TypeError, match="needs dict samples"):
            collate([torch.ones(3) for _ in range(4)])

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for writing a dataset in any format chuchichaestli reads."""

import numpy as np
import pytest
import torch
from chuchichaestli.data import (
    HDF5Dataset,
    NumpyDataset,
    SafetensorsDataset,
    save_dataset,
)
from chuchichaestli.data.save import SAVERS

# every format, what reads back what `save_dataset` wrote to it, and what that
# reader is told when the file carries metadata as well
READERS = {
    ".h5": (HDF5Dataset, {"groups": "data"}, {"attrs_groups": "attrs"}),
    ".hdf5": (HDF5Dataset, {"groups": "data"}, {"attrs_groups": "attrs"}),
    ".safetensors": (SafetensorsDataset, {"keys": "data"}, {"attrs_keys": "attrs"}),
    ".npy": (NumpyDataset, {}, {"attrs_keys": "attrs"}),
    ".npz": (NumpyDataset, {}, {"attrs_keys": "attrs"}),
}


@pytest.fixture
def samples() -> torch.Tensor:
    """Ten samples of a shape with no square axes, to catch a transposition."""
    return torch.randn(10, 1, 4, 6, generator=torch.Generator().manual_seed(0))


def read(path, suffix: str, with_attrs: bool = False):
    """Open the dataset that reads this suffix.

    Args:
        path: File to read.
        suffix: Suffix naming the format.
        with_attrs: Whether to have it read the metadata too.
    """
    reader, kwargs, attrs_kwargs = READERS[suffix]
    return reader(str(path), **kwargs, **(attrs_kwargs if with_attrs else {}))


class TestSaveDataset:
    """Writing samples out, in the format the suffix names."""

    @pytest.mark.parametrize("suffix", sorted(READERS))
    def test_round_trips_through_its_reader(self, tmp_path, samples, suffix):
        """What is written to a suffix is what the dataset reading it holds."""
        path = save_dataset(tmp_path / f"samples{suffix}", samples)
        dataset = read(path, suffix)
        try:
            assert len(dataset) == 10
            assert tuple(dataset.sample_shape) == (1, 4, 6)
            assert torch.equal(dataset[0], samples[0])
            assert torch.equal(dataset[9], samples[9])
        finally:
            dataset.close()

    @pytest.mark.parametrize("suffix", sorted(SAVERS))
    def test_every_saver_writes_a_file(self, tmp_path, samples, suffix):
        """Every suffix the table names is one a caller can actually write."""
        path = save_dataset(tmp_path / f"samples{suffix}", samples)
        assert path.is_file() and path.stat().st_size > 0

    def test_returns_the_path_written(self, tmp_path, samples):
        """The path comes back, so a caller can size or move what it wrote."""
        path = save_dataset(tmp_path / "samples.h5", samples)
        assert path == tmp_path / "samples.h5"

    def test_takes_a_string_path(self, tmp_path, samples):
        """A path is a `str` as readily as a `Path`."""
        assert save_dataset(str(tmp_path / "samples.npy"), samples).is_file()

    def test_creates_the_directory(self, tmp_path, samples):
        """A caller writing into a fresh tree does not have to make it first."""
        path = save_dataset(tmp_path / "a" / "b" / "samples.npy", samples)
        assert path.is_file()

    def test_takes_a_numpy_array(self, tmp_path, samples):
        """Samples are as readily an array as a tensor."""
        path = save_dataset(tmp_path / "samples.h5", samples.numpy())
        dataset = read(path, ".h5")
        try:
            assert torch.equal(dataset[0], samples[0])
        finally:
            dataset.close()

    def test_key_names_the_samples(self, tmp_path, samples):
        """The key a format names its samples by is the one read back."""
        path = save_dataset(tmp_path / "samples.h5", samples, key="images")
        dataset = HDF5Dataset(str(path), groups="images")
        try:
            assert torch.equal(dataset[0], samples[0])
        finally:
            dataset.close()

    def test_key_is_written_into_an_archive(self, tmp_path, samples):
        """An `.npz` names its array, so what was asked for is what is in it."""
        path = save_dataset(tmp_path / "samples.npz", samples, key="images")
        with np.load(path) as archive:
            assert list(archive) == ["images"]

    def test_dtype_survives(self, tmp_path):
        """A format holds the samples it was given, not a promoted copy."""
        path = save_dataset(tmp_path / "s.h5", torch.zeros(4, 2, dtype=torch.float64))
        dataset = HDF5Dataset(str(path), groups="data", dtype=torch.float64)
        try:
            assert dataset[0].dtype is torch.float64
        finally:
            dataset.close()

    def test_overwrites(self, tmp_path, samples):
        """Writing twice leaves the second dataset, not both."""
        save_dataset(tmp_path / "samples.h5", samples)
        path = save_dataset(tmp_path / "samples.h5", samples[:4])
        dataset = read(path, ".h5")
        try:
            assert len(dataset) == 4
        finally:
            dataset.close()

    @pytest.mark.parametrize("suffix", sorted(SAVERS))
    def test_writes_a_tensor_that_requires_grad(self, tmp_path, samples, suffix):
        """Samples straight out of a model are written, not refused."""
        data = samples.clone().requires_grad_(True)
        assert save_dataset(tmp_path / f"samples{suffix}", data).is_file()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a gpu")
    @pytest.mark.parametrize("suffix", sorted(SAVERS))
    def test_writes_a_tensor_on_a_device(self, tmp_path, samples, suffix):
        """Samples living on a gpu are brought home, not refused."""
        path = save_dataset(tmp_path / f"samples{suffix}", samples.cuda())
        assert path.is_file()

    def test_unknown_suffix_raises(self, tmp_path, samples):
        """A suffix no dataset reads fails by name, listing what does."""
        with pytest.raises(ValueError, match="Cannot write '.parquet'"):
            save_dataset(tmp_path / "samples.parquet", samples)

    def test_suffixless_path_raises(self, tmp_path, samples):
        """A path naming no format is refused rather than guessed at."""
        with pytest.raises(ValueError, match="Cannot write ''"):
            save_dataset(tmp_path / "samples", samples)


class TestMetadata:
    """Writing metadata where the dataset reading it looks."""

    @pytest.mark.parametrize("suffix", sorted(READERS))
    def test_numbers_per_sample_round_trip(self, tmp_path, samples, suffix):
        """One number per sample comes back with the sample it belongs to."""
        attrs = np.arange(10.0) * 10
        path = save_dataset(tmp_path / f"samples{suffix}", samples, attrs=attrs)
        dataset = read(path, suffix, with_attrs=True)
        try:
            item, meta = dataset[3]
            assert torch.equal(item, samples[3])
            assert float(meta) == 30.0
        finally:
            dataset.close()

    @pytest.mark.parametrize("suffix", sorted(READERS))
    def test_no_metadata_is_a_bare_sample(self, tmp_path, samples, suffix):
        """A file written without metadata reads as samples, not as pairs."""
        path = save_dataset(tmp_path / f"samples{suffix}", samples)
        dataset = read(path, suffix, with_attrs=True)
        try:
            assert torch.equal(dataset[3], samples[3])
        finally:
            dataset.close()

    def test_a_mapping_is_metadata_of_the_dataset(self, tmp_path, samples):
        """HDF5 has attributes of its own, so a mapping comes back a mapping."""
        attrs = {"instrument": "hst", "pixels": 4}
        path = save_dataset(tmp_path / "samples.h5", samples, attrs=attrs)
        dataset = read(path, ".h5", with_attrs=True)
        try:
            _, meta = dataset[3]
            assert meta == {"instrument": "hst", "pixels": 4}
        finally:
            dataset.close()

    def test_anything_picklable_per_sample_in_a_sidecar(self, tmp_path, samples):
        """The `.npy` sidecar is pickled, so a dict per sample round-trips."""
        attrs = [{"i": i} for i in range(10)]
        path = save_dataset(tmp_path / "samples.npy", samples, attrs=attrs)
        dataset = read(path, ".npy", with_attrs=True)
        try:
            assert dataset[3][1] == {"i": 3}
        finally:
            dataset.close()

    def test_sidecar_is_named_beside_the_dataset(self, tmp_path, samples):
        """`.npy` metadata lands in the sidecar the reader looks for."""
        save_dataset(tmp_path / "samples.npy", samples, attrs=np.arange(10.0))
        assert (tmp_path / "samples.attrs.npy").is_file()

    def test_attrs_key_names_the_sidecar(self, tmp_path, samples):
        """The key is the sidecar's suffix, so a reader can be told another."""
        path = save_dataset(
            tmp_path / "samples.npy", samples, attrs=np.arange(10.0), attrs_key="meta"
        )
        assert (tmp_path / "samples.meta.npy").is_file()
        dataset = NumpyDataset(str(path), attrs_keys="meta", attrs_suffix=".meta")
        try:
            assert float(dataset[3][1]) == 3.0
        finally:
            dataset.close()

    def test_attrs_key_names_the_metadata(self, tmp_path, samples):
        """Elsewhere the key names the group, tensor or array it is written as."""
        path = save_dataset(
            tmp_path / "samples.h5", samples, attrs=np.arange(10.0), attrs_key="meta"
        )
        dataset = HDF5Dataset(str(path), groups="data", attrs_groups="meta")
        try:
            assert float(dataset[3][1]) == 3.0
        finally:
            dataset.close()

    @pytest.mark.parametrize("suffix", [".safetensors", ".npy", ".npz"])
    def test_a_mapping_is_refused_outside_hdf5(self, tmp_path, samples, suffix):
        """No other format holds dataset-wide metadata, so it says so."""
        with pytest.raises(ValueError, match="Cannot write a mapping"):
            save_dataset(tmp_path / f"s{suffix}", samples, attrs={"a": 1})

    @pytest.mark.parametrize("suffix", [".h5", ".safetensors", ".npz"])
    def test_unpicklable_formats_refuse_objects(self, tmp_path, samples, suffix):
        """Metadata a format is read back without pickling is refused."""
        attrs = [{"i": i} for i in range(10)]
        with pytest.raises(ValueError, match="read back without pickling"):
            save_dataset(tmp_path / f"s{suffix}", samples, attrs=attrs)

    @pytest.mark.parametrize(
        "suffix,attrs",
        [(".h5", [{"i": 0}]), (".safetensors", {"a": 1}), (".npz", {"a": 1})],
    )
    def test_a_refusal_writes_nothing(self, tmp_path, samples, suffix, attrs):
        """Metadata a format cannot hold costs the caller no half-written file."""
        with pytest.raises(ValueError):
            save_dataset(tmp_path / f"s{suffix}", samples, attrs=attrs)
        assert list(tmp_path.iterdir()) == []


class TestAtomicity:
    """Landing every file a write touches, or none of them."""

    @pytest.mark.parametrize("suffix", sorted(SAVERS))
    def test_no_scratch_is_left_behind(self, tmp_path, samples, suffix):
        """A finished write leaves the dataset and nothing else."""
        save_dataset(tmp_path / f"samples{suffix}", samples)
        assert [p.name for p in tmp_path.iterdir()] == [f"samples{suffix}"]

    def test_no_scratch_is_left_beside_a_sidecar(self, tmp_path, samples):
        """The two files a `.npy` with metadata writes both land named."""
        save_dataset(tmp_path / "samples.npy", samples, attrs=np.arange(10.0))
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "samples.attrs.npy",
            "samples.npy",
        ]

    def test_a_failed_write_leaves_no_file(self, tmp_path):
        """A write that raises part way through leaves nothing to be read."""
        with pytest.raises(TypeError):
            save_dataset(tmp_path / "samples.h5", [{"not": "an array"}])
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_rewrite_leaves_the_old_dataset(self, tmp_path, samples):
        """The dataset a sweep already has survives a rewrite that fails."""
        path = save_dataset(tmp_path / "samples.h5", samples)
        with pytest.raises(TypeError):
            save_dataset(path, [{"not": "an array"}])
        dataset = read(path, ".h5")
        try:
            assert len(dataset) == 10
            assert torch.equal(dataset[9], samples[9])
        finally:
            dataset.close()
        assert [p.name for p in tmp_path.iterdir()] == ["samples.h5"]

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for writing a dataset that grows, one batch at a time."""

import struct

import h5py
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from chuchichaestli.data import (
    BufferedArchive,
    Hdf5Archive,
    archive_for,
)
from chuchichaestli.data.archive import NpyArchive, SafetensorsArchive
from chuchichaestli.data.archive import DATASETS, merge_archives, read_archive


def test_hdf5_appends_as_it_goes(tmp_path):
    """Each batch reaches the file without the others being held.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.h5"
    with Hdf5Archive(path) as archive:
        for value in range(3):
            archive.write(torch.full((4, 2), float(value)))
    with h5py.File(path) as handle:
        stored = handle["data"][:]
    assert stored.shape == (12, 2)
    assert np.array_equal(stored[0], np.zeros(2))
    assert np.array_equal(stored[-1], np.full(2, 2.0))


def test_a_buffered_archive_writes_the_same_thing(tmp_path):
    """The fallback differs in when it writes, not in what.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    streamed, buffered = tmp_path / "a.h5", tmp_path / "b.npy"
    batches = [torch.rand(4, 2) for _ in range(3)]
    with Hdf5Archive(streamed) as archive:
        for batch in batches:
            archive.write(batch)
    with BufferedArchive(buffered) as archive:
        for batch in batches:
            archive.write(batch)
    with h5py.File(streamed) as handle:
        assert np.allclose(handle["data"][:], np.load(buffered))


def test_an_archive_lands_only_when_closed(tmp_path):
    """A half-written archive leaves no file to mistake for a whole one.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.h5"
    archive = Hdf5Archive(path)
    archive.write(torch.ones(4, 2))
    assert not path.exists()
    archive.close()
    assert path.exists()


def test_an_aborted_archive_leaves_nothing(tmp_path):
    """Neither the file nor its scratch survives.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.h5"
    archive = Hdf5Archive(path)
    archive.write(torch.ones(4, 2))
    archive.abort()
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_a_failing_block_aborts_the_archive(tmp_path):
    """Used as a context manager, an exception discards the file.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.h5"
    with pytest.raises(RuntimeError, match="boom"):
        with Hdf5Archive(path) as archive:
            archive.write(torch.ones(4, 2))
            raise RuntimeError("boom")
    assert not path.exists()


def test_the_suffix_picks_the_archive(tmp_path):
    """Appendable formats stream; the rest fall back to buffering.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    assert isinstance(archive_for(tmp_path / "a.h5"), Hdf5Archive)
    assert archive_for(tmp_path / "a.h5").appends
    assert isinstance(archive_for(tmp_path / "a.npy"), NpyArchive)
    assert archive_for(tmp_path / "a.npy").appends
    assert isinstance(archive_for(tmp_path / "a.safetensors"), SafetensorsArchive)
    assert archive_for(tmp_path / "a.safetensors").appends
    with pytest.warns(UserWarning, match="cannot be appended to"):
        fallback = archive_for(tmp_path / "a.npz")
    assert isinstance(fallback, BufferedArchive)
    assert not fallback.appends


def test_the_fallback_warning_can_be_silenced(tmp_path):
    """A caller that knows what it chose need not be told.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        archive_for(tmp_path / "a.npz", warn=False)


def test_an_empty_archive_writes_nothing(tmp_path):
    """A stage that produced nothing leaves no file behind.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    BufferedArchive(path).close()
    assert not path.exists()


DTYPES = [torch.float32, torch.float64, torch.uint8, torch.int64, torch.bool]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("suffix", sorted(DATASETS))
def test_every_format_round_trips_what_was_written(suffix, dtype, tmp_path):
    """What comes back must equal what went in, to the bit and to the type.

    Args:
        suffix: Extension under test.
        dtype: Type the samples are written as.
        tmp_path: Directory pytest gives the test.
    """
    torch.manual_seed(0)
    source = (torch.rand(10, 3, 4) * 10).to(dtype)
    path = tmp_path / f"out{suffix}"
    with archive_for(path, "data", warn=False) as archive:
        for start in range(0, len(source), 4):
            archive.write(source[start : start + 4])
    stored = torch.cat(list(read_archive(path, "data", dtype=dtype)))
    assert stored.dtype == source.dtype
    assert torch.equal(stored, source)


@pytest.mark.parametrize("suffix", sorted(DATASETS))
def test_reading_gives_the_type_it_was_asked_for(suffix, tmp_path):
    """A dataset names its own type rather than inheriting the file's.

    Args:
        suffix: Extension under test.
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / f"out{suffix}"
    with archive_for(path, "data", warn=False) as archive:
        archive.write(torch.arange(6, dtype=torch.uint8).reshape(3, 2))
    assert next(read_archive(path, "data")).dtype == torch.float32
    assert next(read_archive(path, "data", dtype=torch.uint8)).dtype == torch.uint8


@pytest.mark.parametrize("suffix", sorted(DATASETS))
def test_every_format_joins_into_one(suffix, tmp_path):
    """Merging is what a sharded prediction relies on, for any format.

    Args:
        suffix: Extension under test.
        tmp_path: Directory pytest gives the test.
    """
    shards = []
    for shard in range(2):
        path = tmp_path / f"part{shard}{suffix}"
        with archive_for(path, "data", warn=False) as archive:
            archive.write(torch.arange(6.0).reshape(3, 2) + shard * 100)
        shards.append(path)
    merged = merge_archives(shards, tmp_path / f"all{suffix}", "data")
    stored = torch.cat(list(read_archive(merged, "data")))
    expected = torch.cat(
        [torch.arange(6.0).reshape(3, 2) + shard * 100 for shard in range(2)]
    )
    assert torch.equal(stored, expected)


def test_an_unreadable_format_says_which_are_readable(tmp_path):
    """The suffix is what picks the reader, so a bad one names the rest."""
    with pytest.raises(ValueError, match="archive format"):
        next(read_archive(tmp_path / "out.parquet", "data"))


def test_npy_appends_as_it_goes(tmp_path):
    """Each batch reaches the file without the others being held.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    with NpyArchive(path) as archive:
        for value in range(3):
            archive.write(torch.full((4, 2), float(value)))
    stored = np.load(path)
    assert stored.shape == (12, 2)
    assert stored[:, 0].tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4


def test_the_npy_row_count_is_written_in_without_moving_the_data(tmp_path):
    """The header is reserved at full width, so patching it cannot shift rows.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    with NpyArchive(path) as archive:
        archive.write(torch.arange(4.0).reshape(2, 2))
    with open(path, "rb") as handle:
        assert np.lib.format.read_magic(handle) == (2, 0)
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        assert handle.tell() == 128
    assert shape == (2, 2)
    assert not fortran
    assert dtype == np.dtype("float32")


def test_an_npy_archive_keeps_the_dtype_it_was_given(tmp_path):
    """Streaming writes raw bytes, so nothing silently widens them.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    with NpyArchive(path) as archive:
        archive.write(torch.arange(4, dtype=torch.uint8).reshape(2, 2))
    assert np.load(path).dtype == np.uint8


def test_an_empty_npy_archive_writes_nothing(tmp_path):
    """A stage that produced nothing leaves no file behind.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    NpyArchive(path).close()
    assert not path.exists()


def test_an_aborted_npy_archive_leaves_nothing(tmp_path):
    """A run that failed partway must not leave a half-written file.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    archive = NpyArchive(path)
    archive.write(torch.zeros(2, 2))
    archive.abort()
    assert not path.exists()


def test_safetensors_appends_as_it_goes(tmp_path):
    """Each batch reaches the file without the others being held.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.safetensors"
    with SafetensorsArchive(path, "preds") as archive:
        for value in range(3):
            archive.write(torch.full((4, 2), float(value)))
    stored = load_file(str(path))["preds"]
    assert tuple(stored.shape) == (12, 2)
    assert stored[:, 0].tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4


def test_the_safetensors_header_leaves_the_data_aligned(tmp_path):
    """The format asks for it, and padding the header is what delivers it.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.safetensors"
    with SafetensorsArchive(path) as archive:
        archive.write(torch.arange(4.0).reshape(2, 2))
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
    assert (8 + length) % 8 == 0


def test_a_safetensors_archive_keeps_the_dtype_it_was_given(tmp_path):
    """Streaming writes raw bytes, so nothing silently widens them.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.safetensors"
    with SafetensorsArchive(path) as archive:
        archive.write(torch.arange(4, dtype=torch.uint8).reshape(2, 2))
    assert load_file(str(path))["data"].dtype == torch.uint8


def test_a_type_safetensors_cannot_name_is_refused(tmp_path):
    """Better than writing a header no reader can make sense of.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    with pytest.raises(ValueError, match="safetensors dtype"):
        SafetensorsArchive(tmp_path / "out.safetensors").write(
            torch.ones(2, 2, dtype=torch.complex64)
        )


def test_an_empty_safetensors_archive_writes_nothing(tmp_path):
    """A stage that produced nothing leaves no file behind.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.safetensors"
    SafetensorsArchive(path).close()
    assert not path.exists()

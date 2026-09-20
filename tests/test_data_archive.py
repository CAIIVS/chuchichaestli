# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for writing a dataset that grows, one batch at a time."""

import h5py
import numpy as np
import pytest
import torch

from chuchichaestli.data import (
    BufferedArchive,
    Hdf5Archive,
    archive_for,
)


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
    with pytest.warns(UserWarning, match="cannot be appended to"):
        fallback = archive_for(tmp_path / "a.npy")
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
        archive_for(tmp_path / "a.npy", warn=False)


def test_an_empty_archive_writes_nothing(tmp_path):
    """A stage that produced nothing leaves no file behind.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    path = tmp_path / "out.npy"
    BufferedArchive(path).close()
    assert not path.exists()

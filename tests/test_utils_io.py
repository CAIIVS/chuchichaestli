# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the filesystem helpers shared by the data and programs layers."""

import pytest
import torch

from chuchichaestli.utils.io import (
    READERS,
    METADATA_KEY,
    WRITERS,
    read_state,
    reader_for,
    staged,
    write_state,
    writer_for,
)


STATE = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}


@pytest.mark.parametrize("suffix", sorted(READERS))
def test_state_round_trips_in_every_format(tmp_path, suffix):
    """What one format writes, the matching reader must return unchanged."""
    path = tmp_path / f"state{suffix}"
    write_state(path, STATE)
    assert torch.equal(read_state(path)["weight"], STATE["weight"])


def test_readers_and_writers_cover_the_same_formats():
    """A format that can be written but not read would be a trap."""
    assert set(READERS) == set(WRITERS)


def test_a_torch_archive_is_read_without_unpickling_objects(tmp_path):
    """Loading a foreign checkpoint must not be able to execute code."""
    path = tmp_path / "state.pt"
    torch.save({"weight": STATE["weight"], "extra": object()}, path)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|unpickl"):
        read_state(path)


@pytest.mark.parametrize(
    ("verb", "lookup"), [("read", reader_for), ("write", writer_for)]
)
def test_an_unknown_suffix_names_the_alternatives(tmp_path, verb, lookup):
    """The message follows the package's usual shape."""
    with pytest.raises(ValueError, match=f"Cannot {verb} '.ckpt'"):
        lookup(tmp_path / "x.ckpt")


def test_the_lookups_return_the_handler(tmp_path):
    """Callers that must check before acting get the handler back."""
    assert writer_for(tmp_path / "x.pt") is WRITERS[".pt"]
    assert reader_for(tmp_path / "x.pt") is READERS[".pt"]


def test_staged_moves_into_place_only_on_success(tmp_path):
    """An interrupted write must leave the previous file untouched."""
    target = tmp_path / "out.txt"
    target.write_text("old")
    with pytest.raises(RuntimeError):
        with staged(target) as (scratch,):
            scratch.write_text("new")
            raise RuntimeError("interrupted")
    assert target.read_text() == "old"
    assert not list(tmp_path.glob("*.part*"))

    with staged(target) as (scratch,):
        scratch.write_text("new")
    assert target.read_text() == "new"


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_a_stored_spec_never_reaches_the_state_dict(tmp_path, suffix):
    """A reserved key beside the tensors must not look like a parameter."""
    path = tmp_path / f"model{suffix}"
    write_state(path, STATE, spec='{"cls": "x", "kwargs": {}}')
    assert set(read_state(path)) == set(STATE)
    assert METADATA_KEY not in read_state(path)


def test_a_move_failing_part_way_leaves_no_scratch(tmp_path, monkeypatch):
    """The files not yet moved must not be left behind as stray scratch."""
    from pathlib import Path

    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    replace = Path.replace

    def refuse_second(self, target):
        """Move the first scratch into place, then fail.

        Args:
            self: The scratch being moved.
            target: Where the scratch is moved to.
        """
        if Path(target) == second:
            raise OSError("disk full")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", refuse_second)
    with pytest.raises(OSError, match="disk full"):
        with staged(first, second) as scratches:
            for scratch in scratches:
                scratch.write_text("new")
    assert first.read_text() == "new"
    assert not list(tmp_path.glob("*.part*"))

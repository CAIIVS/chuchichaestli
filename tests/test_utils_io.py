# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the filesystem helpers shared by the data and runtime layers."""

import pytest
import torch

from chuchichaestli.models.spec import InitArgMixin
from chuchichaestli.utils.io import (
    READERS,
    SPEC_KEY,
    load_model,
    read_spec,
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


class Tiny(InitArgMixin, torch.nn.Module):
    """A model whose activation leaves no tensors behind."""

    def __init__(self, width: int = 4, act: str = "silu"):
        """Constructor.

        Args:
            width: Width of the linear layer.
            act: Which activation to use.
        """
        super().__init__()
        self.lin = torch.nn.Linear(width, width)
        self.act = {"silu": torch.nn.SiLU, "gelu": torch.nn.GELU}[act]()


def test_a_spec_travels_inside_the_weights(tmp_path):
    """The whole point: the file says which architecture built it."""
    path = tmp_path / "model.safetensors"
    model = Tiny(width=8, act="gelu")
    write_state(path, model.state_dict(), spec=model.spec)

    rebuilt = load_model(path)
    assert type(rebuilt.act).__name__ == "GELU"
    assert torch.equal(rebuilt.lin.weight, model.lin.weight)


def test_weights_without_a_spec_refuse_to_guess(tmp_path):
    """Inferring the architecture would load cleanly and compute the wrong thing."""
    path = tmp_path / "plain.safetensors"
    write_state(path, Tiny().state_dict())
    assert read_spec(path) is None
    with pytest.raises(ValueError, match="No model spec"):
        load_model(path)


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_a_spec_survives_every_format(tmp_path, suffix):
    """A torch archive has no metadata header, but it can still carry one."""
    path = tmp_path / f"model{suffix}"
    model = Tiny(width=8, act="gelu")
    write_state(path, model.state_dict(), spec=model.spec)
    rebuilt = load_model(path)
    assert type(rebuilt.act).__name__ == "GELU"
    assert torch.equal(rebuilt.lin.weight, model.lin.weight)


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_the_spec_never_reaches_the_state_dict(tmp_path, suffix):
    """A reserved key beside the tensors must not look like a parameter."""
    path = tmp_path / f"model{suffix}"
    model = Tiny()
    write_state(path, model.state_dict(), spec=model.spec)
    assert set(read_state(path)) == set(model.state_dict())
    assert SPEC_KEY not in read_state(path)


def test_an_unknown_suffix_cannot_hold_a_spec_either(tmp_path):
    """The refusal names the formats that do work."""
    model = Tiny()
    with pytest.raises(ValueError, match="Cannot write '.ckpt'"):
        write_state(tmp_path / "model.ckpt", model.state_dict(), spec=model.spec)


def test_load_model_accepts_overrides(tmp_path):
    """Swapping a parameter-free choice should not need the spec edited by hand."""
    path = tmp_path / "model.safetensors"
    model = Tiny(width=4, act="gelu")
    write_state(path, model.state_dict(), spec=model.spec)
    rebuilt = load_model(path, act="silu")
    assert type(rebuilt.act).__name__ == "SiLU"
    assert torch.equal(rebuilt.lin.weight, model.lin.weight)


def test_an_override_that_changes_shapes_fails_loudly(tmp_path):
    """`strict` covers key names, not shapes, so this must still raise."""
    path = tmp_path / "model.safetensors"
    model = Tiny(width=4)
    write_state(path, model.state_dict(), spec=model.spec)
    with pytest.raises(RuntimeError, match="(?i)size mismatch|shape"):
        load_model(path, width=6)

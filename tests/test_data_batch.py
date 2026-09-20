# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for reading named values out of a batch."""

import pytest
import torch

from chuchichaestli.data import batch_to_device, samples_in_batch, unpack_batch


def test_a_mapping_is_read_by_name():
    """Names select values, in the order they were asked for."""
    x, y = torch.ones(4, 2), torch.zeros(4, 2)
    assert unpack_batch({"x": x, "y": y}, "x", "y") == (x, y)
    assert unpack_batch({"x": x, "y": y}, "y", "x") == (y, x)


def test_a_sequence_is_read_by_position():
    """A tuple batch is taken in order."""
    x, y = torch.ones(4, 2), torch.zeros(4, 2)
    assert unpack_batch((x, y), "x", "y") == (x, y)


def test_any_number_of_names_is_read():
    """The reader is not limited to an input and a target."""
    batch = {"x": 1, "y": 2, "mask": 3}
    assert unpack_batch(batch, "x") == (1,)
    assert unpack_batch(batch, "x", "y", "mask") == (1, 2, 3)


def test_a_missing_key_is_named():
    """A batch lacking a requested name says which."""
    with pytest.raises(ValueError, match=r"no \['mask'\]"):
        unpack_batch({"x": 1, "y": 2}, "x", "mask")


def test_a_sequence_of_the_wrong_length_is_refused():
    """A positional batch must hold one value per name."""
    with pytest.raises(ValueError, match="reads 2 values, got 3 items"):
        unpack_batch((1, 2, 3), "x", "y")


def test_a_batch_that_is_neither_is_refused():
    """A bare object cannot be read positionally or by name."""
    with pytest.raises(ValueError, match="got a object"):
        unpack_batch(object(), "x", "y")


def test_the_reader_names_itself_in_errors():
    """The caller's name carries into the message, defaulting when absent."""
    with pytest.raises(ValueError, match="Batch reader reads"):
        unpack_batch((1, 2, 3), "x", "y")
    with pytest.raises(ValueError, match="Reconstruction reads"):
        unpack_batch((1, 2, 3), "x", "y", reader="Reconstruction")


def test_no_names_is_refused():
    """Asking for nothing is a mistake, not an empty tuple."""
    with pytest.raises(ValueError, match="at least one name"):
        unpack_batch({"x": 1})


def test_samples_in_batch_finds_the_batch_size():
    """A batch is measured through whatever tensor it holds."""
    assert samples_in_batch(torch.ones(7, 3)) == 7
    assert samples_in_batch({"x": torch.ones(5, 2)}) == 5
    assert samples_in_batch((torch.ones(4), torch.ones(4))) == 4


def test_samples_in_batch_falls_back_to_one():
    """A batch holding no tensor counts as a single sample."""
    assert samples_in_batch(object()) == 1
    assert samples_in_batch({"meta": "no tensors here"}) == 1


def test_a_batch_moves_to_a_device_whatever_shape_it_has():
    """Every tensor moves, and the container keeps its shape."""
    pair = batch_to_device((torch.ones(2, 2), torch.zeros(2, 1)), "cpu")
    assert isinstance(pair, tuple) and all(t.device.type == "cpu" for t in pair)
    mapping = batch_to_device({"x": torch.ones(2, 2)}, "cpu")
    assert isinstance(mapping, dict) and mapping["x"].device.type == "cpu"

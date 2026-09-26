# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for reading named values out of a batch."""

import pytest
import torch

from chuchichaestli.data import (
    as_image_batch,
    batch_to_device,
    input_in_batch,
    samples_in_batch,
    unpack_batch,
)


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


def test_the_leading_value_ignores_what_else_a_batch_carries():
    """A batch holding a target alongside its input still yields the input."""
    x, y = torch.ones(2, 3), torch.zeros(2, 1)
    assert input_in_batch({"x": x, "y": y}) is x
    assert input_in_batch((x, y)) is x
    assert input_in_batch(x) is x
    assert input_in_batch({"img": x}, "img") is x


def test_the_leading_value_is_stricter_than_nothing():
    """A mapping without the key, or an empty batch, says so."""
    with pytest.raises(ValueError, match="no 'x' to read"):
        input_in_batch({"a": torch.ones(2)})
    with pytest.raises(ValueError, match="leads with nothing"):
        input_in_batch(())


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((16, 16), (1, 1, 16, 16)),
        ((3, 16, 16), (1, 3, 16, 16)),
        ((16, 16, 4), (1, 4, 16, 16)),
        ((8, 1, 12, 12), (8, 1, 12, 12)),
        ((8, 12, 12, 3), (8, 3, 12, 12)),
        ((8, 12, 12), (8, 1, 12, 12)),
    ],
)
def test_every_image_layout_becomes_a_channels_first_batch(shape, expected):
    """Channels first or last, one image or many, all end up the same."""
    assert tuple(as_image_batch(torch.rand(*shape)).shape) == expected


def test_an_image_batch_promotes_integers():
    """Pixels read off disk as `uint8` still have to scale and plot."""
    assert as_image_batch(torch.randint(0, 255, (4, 8, 8))).is_floating_point()


def test_a_shape_that_is_no_image_says_so():
    """Five channels is not a picture, and the error has to say why."""
    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        as_image_batch(torch.rand(2, 5, 8, 8))

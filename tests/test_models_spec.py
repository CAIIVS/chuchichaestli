# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for recording how an object was constructed."""

import pytest
import torch
from torch import nn

from chuchichaestli.utils.functools import partialclass
from chuchichaestli.models.spec import (
    InitArgMixin,
    ModelSpec,
    qualname,
    render,
    resolve,
)


class Tiny(InitArgMixin, nn.Module):
    """A model whose choices mostly leave no tensors behind."""

    def __init__(
        self, width: int = 4, act: str = "silu", dropout: float = 0.0, **extra
    ):
        """Constructor.

        Args:
            width: Width of the linear layer.
            act: Which activation to use.
            dropout: Dropout probability.
            extra: Ignored, present to exercise variadic capture.
        """
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.act = {"silu": nn.SiLU, "gelu": nn.GELU}[act]()
        self.drop = nn.Dropout(dropout)


Preset = partialclass("Preset", Tiny, 8, act="gelu")


def test_every_argument_is_recorded_including_defaults():
    """`.spec` must be complete, not just what the caller happened to pass."""
    spec = Tiny(width=8).spec
    assert spec.kwargs == {"width": 8, "act": "silu", "dropout": 0.0}


def test_variadic_keywords_are_flattened():
    """A `**kwargs` surface must not nest inside the recorded arguments."""
    assert Tiny(width=2, tag="a").spec.kwargs["tag"] == "a"


def test_a_partialclass_records_the_bound_arguments():
    """The repo builds variants this way, so they must record fully."""
    spec = Preset().spec
    assert spec.kwargs["width"] == 8
    assert spec.kwargs["act"] == "gelu"
    assert spec.cls.endswith(":Preset")


def test_rebuilding_restores_what_weights_cannot_carry():
    """Activation and dropout leave no tensors, so only the spec preserves them."""
    original = Tiny(width=8, act="gelu", dropout=0.3)
    rebuilt = original.spec.build()
    assert type(rebuilt.act).__name__ == "GELU"
    assert rebuilt.drop.p == 0.3
    assert rebuilt.lin.in_features == 8


def test_build_accepts_overrides():
    """Rebuilding a variant should not need the spec editing by hand."""
    assert Tiny(width=4).spec.build(width=6).lin.in_features == 6


def test_spec_survives_json():
    """The spec travels as one string in file metadata."""
    spec = Tiny(width=8, act="gelu").spec
    assert ModelSpec.from_json(spec.to_json()) == spec


def test_recording_leaves_the_module_intact():
    """Wrapping `__init__` must not disturb parameter registration."""
    model = Tiny(width=8)
    assert len(list(model.parameters())) == 2
    assert isinstance(model, nn.Module)


def test_a_class_without_its_own_init_says_so():
    """Better an explicit error than a spec that silently omits everything."""

    class Bare(InitArgMixin):
        """A class that defines no constructor."""

    with pytest.raises(AttributeError, match="recorded no constructor arguments"):
        Bare().spec


def test_qualname_round_trips_through_resolve():
    """A spec names a class by an importable path."""
    assert resolve(qualname(nn.Linear)) is nn.Linear


def test_resolve_rejects_a_name_that_is_not_importable():
    """The error says what shape the name should have."""
    with pytest.raises(ValueError, match="Use 'module:QualName'"):
        resolve("torch.nn.Linear")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.float32, "float32"),
        (torch.device("cpu"), "cpu"),
        ((1, 2, 3), [1, 2, 3]),
        ({"a": (1, 2)}, {"a": [1, 2]}),
        (None, None),
        (True, True),
    ],
)
def test_render_reduces_arguments_to_json(value, expected):
    """Constructor arguments hold torch types that JSON cannot."""
    assert render(value) == expected

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for recording how an object was constructed."""

import pytest
import torch
from torch import nn

from chuchichaestli.utils.functools import partialclass
from chuchichaestli.models.spec import InitArgMixin, ModelSpec


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


def test_qualname_round_trips_through_import_class():
    """A spec names a class by an importable path."""
    assert ModelSpec.import_class(ModelSpec.qualname(nn.Linear)) is nn.Linear


def test_import_class_rejects_a_name_that_is_not_importable():
    """The error says what shape the name should have."""
    with pytest.raises(ValueError, match="Use 'module:QualName'"):
        ModelSpec.import_class("torch.nn.Linear")


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
def test_encode_arg_reduces_arguments_to_json(value, expected):
    """Constructor arguments hold torch types that JSON cannot."""
    assert ModelSpec.encode_arg(value) == expected


class Wrapper(InitArgMixin, nn.Module):
    """A model assembled from a component it was handed."""

    def __init__(self, inner: nn.Module, scale: float = 1.0):
        """Constructor.

        Args:
            inner: Component to wrap.
            scale: Arbitrary scalar, to check plain arguments still render.
        """
        super().__init__()
        self.inner = inner
        self.scale = scale


def test_a_component_is_recorded_as_a_nested_spec():
    """A model built from submodules must say how to rebuild them too."""
    nested = Wrapper(Tiny(width=8, act="gelu")).spec.to_dict()
    assert nested["kwargs"]["inner"]["cls"].endswith(":Tiny")
    assert nested["kwargs"]["inner"]["kwargs"]["act"] == "gelu"


def test_a_nested_spec_rebuilds_its_component():
    """The whole tree has to survive a trip through JSON."""
    original = Wrapper(Tiny(width=8, act="gelu"), scale=2.0)
    rebuilt = ModelSpec.from_json(original.spec.to_json()).build()
    assert isinstance(rebuilt.inner, Tiny)
    assert type(rebuilt.inner.act).__name__ == "GELU"
    assert rebuilt.inner.lin.in_features == 8
    assert rebuilt.scale == 2.0


def test_a_module_that_records_nothing_is_refused():
    """A repr would serialize fine and then rebuild into the wrong thing."""
    with pytest.raises(TypeError, match="does not inherit InitArgMixin"):
        Wrapper(nn.Linear(4, 4)).spec.to_dict()


def test_decode_arg_passes_plain_values_through():
    """Only spec-shaped mappings are rebuilt; everything else is data."""
    assert ModelSpec.decode_arg({"cls": "x", "kwargs": {}, "extra": 1}) == {
        "cls": "x",
        "kwargs": {},
        "extra": 1,
    }
    assert ModelSpec.decode_arg([1, "a", None]) == [1, "a", None]


def test_recording_does_not_register_the_arguments_as_submodules():
    """A recorded component must not show up twice in the state dict."""
    inner = Tiny(width=4)
    wrapper = Wrapper(inner)
    assert [name for name, _ in wrapper.named_children()] == ["inner"]
    assert all("_init_args" not in key for key in wrapper.state_dict())

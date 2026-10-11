# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for recording how an object was constructed."""

import functools
import math
import sys
from enum import Enum
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from chuchichaestli.utils.functools import partialclass
from chuchichaestli.models.adversarial.discriminator import (
    AntialiasingDiscriminator,
    AntialiasingPatchDiscriminator,
    AttnPatchDiscriminator,
    BlockDiscriminator,
    PatchDiscriminator,
    PixelDiscriminator,
)
from chuchichaestli.models.autoencoder import (
    Autoencoder,
    DCAE,
    Decoder,
    Encoder,
    LiteVAE,
    LiteVAEDecoder,
    LiteVAEEncoder,
    VAE,
    VAEDecoder,
    VAEEncoder,
    VQVAE,
)
from chuchichaestli.models.autoencoder.vqvae import VectorQuantizer
from chuchichaestli.models.resnet import ResNet, ResNet18
from chuchichaestli.models.spec import C3liSpecError, InitArgMixin, ModelSpec
from chuchichaestli.models.unet import UNet


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


@pytest.mark.parametrize("value", [[1, 2, 3], {"a": [1, 2]}, None, True, 1.5, "silu"])
def test_encode_arg_leaves_json_values_unchanged(value):
    """Plain JSON stays readable in the stored spec."""
    assert ModelSpec.encode_arg(value) == value


@ModelSpec.allow
class Mode(Enum):
    """An enum argument, for the round trip."""

    FAST = "fast"


ModelSpec.allow(torch.relu)


def same(x, y):
    """Return whether a rebuilt argument equals the original, type included.

    Args:
        x: Original argument.
        y: Rebuilt argument.
    """
    if type(x) is not type(y):
        return False
    if isinstance(x, torch.Tensor):
        return x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, functools.partial):
        return x.func is y.func and same(x.args, y.args) and x.keywords == y.keywords
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


@pytest.mark.parametrize(
    "value",
    [
        torch.float16,
        torch.device("cpu"),
        (1, (2, 3)),
        {"a": (1, 2)},
        {1: "a", (2, 3): "b"},
        {"cls": 1, "kwargs": 2},
        {"__c3li__": "tensor"},
        Path("runs/a"),
        Mode.FAST,
        nn.GELU,
        F.gelu,
        torch.relu,
        functools.partial(F.leaky_relu, negative_slope=0.1),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
        torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64),
        torch.zeros(0, 3, dtype=torch.int64),
        torch.tensor(True),
    ],
    ids=repr,
)
def test_an_argument_rebuilds_as_itself(value):
    """The rebuilt model receives the argument the original was given."""
    spec = ModelSpec.from_json(Tiny(payload=value).spec.to_json())
    assert same(spec.build().spec.kwargs["payload"], value)


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


def closure(scale):
    """Return a function only reachable through this call.

    Args:
        scale: Factor the returned function multiplies by.
    """

    def scaled(x):
        return scale * x

    return scaled


@pytest.mark.parametrize("value", [lambda x: x, closure(2), object(), b"raw"])
def test_an_argument_with_no_faithful_encoding_is_refused(value):
    """A stand-in would rebuild into something other than the original."""
    with pytest.raises(TypeError, match="Cannot record"):
        Tiny(payload=value).spec.to_dict()


class SpecShaped(InitArgMixin, nn.Module):
    """A model whose arguments share the names of a spec's fields."""

    def __init__(self, cls: str = "x", kwargs: str = "y"):
        """Constructor.

        Args:
            cls: Arbitrary string.
            kwargs: Arbitrary string.
        """
        super().__init__()


def test_arguments_named_like_a_spec_rebuild_as_themselves():
    """Only a nested value can be a nested spec."""
    rebuilt = ModelSpec.from_json(SpecShaped(kwargs="z").spec.to_json()).build()
    assert dict(rebuilt.spec.kwargs) == {"cls": "x", "kwargs": "z"}


@pytest.mark.parametrize(
    "encoded",
    [
        {"cls": "subprocess:run", "kwargs": {"args": ["true"]}},
        {"__c3li__": "import", "name": "subprocess:run"},
        {
            "__c3li__": "partial",
            "func": {"__c3li__": "import", "name": "subprocess:run"},
            "args": [["true"]],
            "kwargs": {},
        },
        {"__c3li__": "enum", "cls": "subprocess:run", "value": ["true"]},
    ],
    ids=["spec", "import", "partial", "enum"],
)
def test_an_untrusted_spec_cannot_name_what_was_not_allowed(encoded):
    """Decoding a file must not call whatever the file names."""
    with pytest.raises(C3liSpecError, match="subprocess:run"):
        ModelSpec.decode_arg(encoded)


def test_a_library_module_does_not_allow_what_it_imports():
    """Only the library's allowed objects resolve through its modules."""
    with pytest.raises(C3liSpecError, match="torch.load"):
        ModelSpec.decode_arg(
            {"__c3li__": "import", "name": "chuchichaestli.utils.io:torch.load"}
        )


def test_a_refused_name_is_not_imported():
    """Importing a module runs its code."""
    sys.modules.pop("this", None)
    with pytest.raises(C3liSpecError):
        ModelSpec.decode_arg({"__c3li__": "import", "name": "this:s"})
    assert "this" not in sys.modules


def test_a_trusted_spec_may_name_anything():
    """A file from a trusted source may name what no one allowed."""
    encoded = {"__c3li__": "import", "name": "math:sqrt"}
    with pytest.raises(C3liSpecError):
        ModelSpec.decode_arg(encoded)
    assert ModelSpec.decode_arg(encoded, trusted=True) is math.sqrt


@pytest.mark.parametrize(
    "encoded",
    [
        {"__c3li__": "dtype", "name": "load"},
        {"__c3li__": "tensor", "dtype": "load", "shape": [1], "data": [0]},
    ],
    ids=["dtype", "tensor"],
)
def test_a_dtype_must_name_a_dtype(encoded):
    """Not every name in the torch namespace is a dtype."""
    with pytest.raises(C3liSpecError, match="as a dtype"):
        ModelSpec.decode_arg(encoded)


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


MODELS = [
    UNet,
    ResNet,
    ResNet18,
    Encoder,
    Decoder,
    Autoencoder,
    VAE,
    VAEEncoder,
    VAEDecoder,
    VQVAE,
    VectorQuantizer,
    DCAE,
    LiteVAE,
    LiteVAEEncoder,
    LiteVAEDecoder,
    BlockDiscriminator,
    PixelDiscriminator,
    PatchDiscriminator,
    AttnPatchDiscriminator,
    AntialiasingDiscriminator,
    AntialiasingPatchDiscriminator,
]


@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_every_model_family_records_its_arguments(model_cls):
    """A model added without the mixin cannot be rebuilt from a checkpoint."""
    assert issubclass(model_cls, InitArgMixin)


def test_a_real_model_rebuilds_from_its_spec():
    """`UNet` keeps two of its forty-four arguments, so the spec is the record."""
    original = UNet(dimensions=2, in_channels=1, n_channels=32)
    rebuilt = ModelSpec.from_json(original.spec.to_json()).build()
    before, after = original.state_dict(), rebuilt.state_dict()
    assert set(before) == set(after)
    assert all(before[key].shape == after[key].shape for key in before)


def test_a_model_assembled_from_components_rebuilds():
    """The autoencoders take their encoder and decoder as constructed objects."""
    original = Autoencoder(
        encoder=Encoder(dimensions=2, in_channels=1, n_channels=32),
        decoder=Decoder(dimensions=2, n_channels=256, out_channels=1),
    )
    rebuilt = ModelSpec.from_json(original.spec.to_json()).build()
    assert isinstance(rebuilt.encoder, Encoder)
    assert isinstance(rebuilt.decoder, Decoder)
    before, after = original.state_dict(), rebuilt.state_dict()
    assert set(before) == set(after)
    assert all(before[key].shape == after[key].shape for key in before)


def test_a_forwarding_constructor_records_what_its_caller_passed():
    """Recording the value the caller passed, not the one forwarded down.

    `VAEEncoder` doubles `out_channels`, so recording the doubled value would
    double it again on every rebuild.
    """
    encoder = VAEEncoder(dimensions=2, in_channels=1, n_channels=32, out_channels=4)
    assert encoder.spec.kwargs["out_channels"] == 4
    rebuilt = ModelSpec.from_json(encoder.spec.to_json()).build()
    assert rebuilt.latent_channels == encoder.latent_channels == 4

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recording how an object was constructed, so it can be rebuilt exactly."""

from __future__ import annotations
import functools
import importlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
import torch
from chuchichaestli.utils.functools import InitArgMixin
from chuchichaestli.utils.io import reader_for, write_state


__all__ = [
    "ModelSpec",
    "ModelSpecMixin",
    "C3liSpecError",
    "read_spec",
    "read_model_state",
    "write_model_state",
    "load_model",
]

_TAG = "__c3li__"
_PACKAGE = __name__.partition(".")[0]


class C3liSpecError(ValueError):
    """Raised when a spec names something it may not load."""


def _names_of(obj: Any) -> list[str]:
    """Return the `"module:QualName"` names that may refer to `obj`.

    Args:
        obj: Class, function or other named object.
    """
    module = getattr(obj, "__module__", None)
    if module is None:
        return []
    names = (
        getattr(obj, attribute, None) for attribute in ("__qualname__", "__name__")
    )
    return [f"{module}:{name}" for name in names if name is not None]


@functools.cache
def _torch_defaults() -> frozenset[Any]:
    """Return the `torch.nn` modules and functional operations allowed by default."""
    modules = (
        value
        for value in vars(torch.nn).values()
        if isinstance(value, type) and issubclass(value, torch.nn.Module)
    )
    operations = (
        value
        for key, value in vars(torch.nn.functional).items()
        if not key.startswith("_") and inspect.isroutine(value)
    )
    return frozenset((*modules, *operations))


def _dtype(name: str) -> torch.dtype:
    """Return the torch dtype called `name`.

    Args:
        name: Name of the dtype, without the `torch.` prefix.

    Raises:
        C3liSpecError: If `name` is not a dtype.
    """
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise C3liSpecError(f"The spec names {name!r} as a dtype, which it is not.")
    return dtype


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What it takes to rebuild a model.

    Attributes:
        cls: Importable name of the class, as `"module:QualName"`.
        kwargs: Every constructor argument, defaults included.
        json_limit: Most characters a spec is written as or read from.
    """

    cls: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    json_limit: ClassVar[int] = 16 << 20
    _allowed: ClassVar[set[Any]] = set()

    @staticmethod
    def qualname(obj: type) -> str:
        """Return an importable `"module:QualName"` for a class.

        Args:
            obj: Class to name.
        """
        return f"{obj.__module__}:{obj.__qualname__}"

    @staticmethod
    def import_class(path: str) -> type:
        """Import the class an importable name points at.

        Args:
            path: Name of the form `"module:QualName"`.

        Raises:
            ValueError: If `path` is not of that form.
        """
        if ":" not in path:
            raise ValueError(
                f"Not an importable name: {path!r}. Use 'module:QualName'."
            )
        module, _, name = path.partition(":")
        target: Any = importlib.import_module(module)
        for part in name.split("."):
            target = getattr(target, part)
        return target

    @staticmethod
    def import_name(obj: Any) -> str:
        """Return a `"module:QualName"` that `import_class` resolves to `obj` itself.

        Args:
            obj: Class, function or other named object.

        Raises:
            TypeError: If no name imports back to `obj`, as for a lambda.
        """
        for path in _names_of(obj):
            try:
                if ModelSpec.import_class(path) is obj:
                    return path
            except (ImportError, AttributeError):
                continue
        raise TypeError(
            f"Cannot record {obj!r} in a spec: no import name resolves to it."
        )

    @classmethod
    def allow(cls, obj: Any) -> Any:
        """Let a spec from an untrusted file name `obj`, and return it.

        Args:
            obj: Class, function or enum to allow.
        """
        cls._allowed.add(obj)
        return obj

    @classmethod
    def resolve(cls, path: str, trusted: bool = False) -> Any:
        """Return the object an import name refers to, if a spec may name it.

        Args:
            path: Name of the form `"module:QualName"`.
            trusted: Import any name, as for a file from a trusted source.

        Raises:
            C3liSpecError: If `path` is not allowed and the file is untrusted.
        """
        if trusted:
            return cls.import_class(path)
        module = path.partition(":")[0]
        if module.partition(".")[0] == _PACKAGE:
            importlib.import_module(module)
        for obj in (*cls._allowed, *_torch_defaults()):
            if path in _names_of(obj):
                return obj
        raise C3liSpecError(
            f"The spec names {path!r}, which is not allowed. Allow it with "
            "ModelSpec.allow(...), or pass trusted=True for a file you trust."
        )

    @classmethod
    def encode_arg(cls, value: Any) -> Any:
        """Reduce a constructor argument to something JSON can hold.

        Args:
            value: Argument to reduce.

        Raises:
            TypeError: If `value` is a module that records no arguments, or
                anything else `decode_arg` could not restore.
        """
        if isinstance(value, ModelSpecMixin):
            return value.spec.to_dict()
        if isinstance(value, torch.nn.Module):
            raise TypeError(
                f"Cannot record a {type(value).__name__} argument in a spec: it "
                "does not inherit ModelSpecMixin, so nothing says how to rebuild it."
            )
        if isinstance(value, torch.Tensor):
            return cls._encode_tensor(value)
        if isinstance(value, torch.dtype):
            return {_TAG: "dtype", "name": str(value).removeprefix("torch.")}
        if isinstance(value, torch.device):
            return {_TAG: "device", "name": str(value)}
        if isinstance(value, Enum):
            return {
                _TAG: "enum",
                "cls": cls.import_name(type(value)),
                "value": cls.encode_arg(value.value),
            }
        if isinstance(value, Path):
            return {_TAG: "path", "path": str(value)}
        if isinstance(value, Mapping):
            return cls._encode_mapping(value)
        if isinstance(value, tuple):
            return {_TAG: "tuple", "items": [cls.encode_arg(v) for v in value]}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls.encode_arg(v) for v in value]
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        if isinstance(value, functools.partial):
            return {
                _TAG: "partial",
                "func": cls.encode_arg(value.func),
                "args": [cls.encode_arg(v) for v in value.args],
                "kwargs": cls.encode_arg(dict(value.keywords)),
            }
        if callable(value):
            return {_TAG: "import", "name": cls.import_name(value)}
        raise TypeError(
            f"Cannot record a {type(value).__name__} argument in a spec: JSON "
            "cannot hold it, so the rebuilt model would receive something else."
        )

    @staticmethod
    def _encode_tensor(value: torch.Tensor) -> dict[str, Any]:
        """Reduce a tensor to its dtype, shape and values.

        Args:
            value: Tensor to reduce.
        """
        tensor = value.detach().cpu()
        data = torch.view_as_real(tensor) if tensor.is_complex() else tensor
        return {
            _TAG: "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": data.tolist(),
        }

    @classmethod
    def _encode_mapping(cls, value: Mapping) -> dict[str, Any]:
        """Reduce a mapping, as key-value pairs if its keys are not plain strings.

        Args:
            value: Mapping to reduce.
        """
        plain = (
            all(isinstance(k, str) for k in value)
            and _TAG not in value
            and set(value) != {"cls", "kwargs"}
        )
        if plain:
            return {k: cls.encode_arg(v) for k, v in value.items()}
        return {
            _TAG: "mapping",
            "items": [[cls.encode_arg(k), cls.encode_arg(v)] for k, v in value.items()],
        }

    @classmethod
    def decode_arg(cls, value: Any, trusted: bool = False) -> Any:
        """Rebuild whatever `encode_arg` reduced, nested specs included.

        Args:
            value: Encoded argument to restore.
            trusted: Import any name, as for a file from a trusted source.
        """
        if isinstance(value, Mapping):
            if _TAG in value:
                return cls._decode_tagged(value, trusted)
            if set(value) == {"cls", "kwargs"}:
                return cls.from_dict(value).build(trusted=trusted)
            return {k: cls.decode_arg(v, trusted) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls.decode_arg(v, trusted) for v in value]
        return value

    @classmethod
    def _decode_tagged(cls, value: Mapping[str, Any], trusted: bool) -> Any:
        """Rebuild an argument `encode_arg` tagged with its kind.

        Args:
            value: Tagged mapping to restore.
            trusted: Import any name, as for a file from a trusted source.

        Raises:
            ValueError: If the tag names no known kind.
        """
        match value[_TAG]:
            case "import":
                return cls.resolve(value["name"], trusted)
            case "partial":
                return functools.partial(
                    cls.decode_arg(value["func"], trusted),
                    *cls.decode_arg(value["args"], trusted),
                    **cls.decode_arg(value["kwargs"], trusted),
                )
            case "tensor":
                dtype = _dtype(value["dtype"])
                if dtype.is_complex:
                    parts = torch.tensor(value["data"], dtype=torch.float64)
                    parts = parts.reshape(*value["shape"], 2)
                    return torch.view_as_complex(parts).to(dtype)
                return torch.tensor(value["data"], dtype=dtype).reshape(value["shape"])
            case "dtype":
                return _dtype(value["name"])
            case "device":
                return torch.device(value["name"])
            case "enum":
                enum = cls.resolve(value["cls"], trusted)
                return enum(cls.decode_arg(value["value"], trusted))
            case "path":
                return Path(value["path"])
            case "tuple":
                return tuple(cls.decode_arg(v, trusted) for v in value["items"])
            case "mapping":
                return {
                    cls.decode_arg(k, trusted): cls.decode_arg(v, trusted)
                    for k, v in value["items"]
                }
        raise ValueError(f"Unknown encoded argument kind {value[_TAG]!r}.")

    def build(self, *, trusted: bool = False, **overrides: Any) -> Any:
        """Construct the object this spec describes.

        Args:
            trusted: Import any name, as for a file from a trusted source.
            overrides: Arguments to replace, for rebuilding a variant.
        """
        target = self.resolve(self.cls, trusted)
        kwargs = {k: self.decode_arg(v, trusted) for k, v in self.kwargs.items()}
        return target(**{**kwargs, **overrides})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the spec."""
        kwargs = {key: self.encode_arg(value) for key, value in self.kwargs.items()}
        return {"cls": self.cls, "kwargs": kwargs}

    def to_json(self) -> str:
        """Return the spec as one JSON string, for file metadata.

        Raises:
            TypeError: If the string is longer than `json_limit`.
        """
        text = json.dumps(self.to_dict(), sort_keys=True)
        if len(text) > self.json_limit:
            raise TypeError(
                f"Cannot record a spec of {len(text)} characters, above "
                f"ModelSpec.json_limit of {self.json_limit}. Register a large "
                "tensor argument as a buffer instead."
            )
        return text

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> ModelSpec:
        """Rebuild a spec from its mapping form.

        Args:
            state: Mapping as returned by `to_dict`.
        """
        return cls(cls=state["cls"], kwargs=dict(state.get("kwargs", {})))

    @classmethod
    def from_json(cls, text: str) -> ModelSpec:
        """Rebuild a spec from its JSON form.

        Args:
            text: String as returned by `to_json`.

        Raises:
            C3liSpecError: If `text` is longer than `json_limit`.
        """
        if len(text) > cls.json_limit:
            raise C3liSpecError(
                f"The spec is {len(text)} characters, above ModelSpec.json_limit "
                f"of {cls.json_limit}. Raise the limit for a file you trust."
            )
        return cls.from_dict(json.loads(text))


class ModelSpecMixin(InitArgMixin):
    """Records every constructor argument, so `.spec` can rebuild the object."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Record a subclass's arguments, and allow it to be built from a spec.

        Args:
            kwargs: Passed to the next `__init_subclass__` in the chain.
        """
        super().__init_subclass__(**kwargs)
        ModelSpec.allow(cls)

    @property
    def spec(self) -> ModelSpec:
        """Return what it takes to rebuild this object.

        Raises:
            AttributeError: If the arguments were never recorded, which means
                the class defines no `__init__` of its own to wrap.
        """
        recorded = self.init_args
        if recorded is None:
            raise AttributeError(
                f"{type(self).__name__} recorded no constructor arguments; "
                "it defines no __init__ for InitArgMixin to wrap."
            )
        return ModelSpec(cls=ModelSpec.qualname(type(self)), kwargs=dict(recorded))


def read_spec(path: Path) -> ModelSpec | None:
    """Return the spec a file carries, or `None` if it carries none.

    Args:
        path: File to read.
    """
    stored = reader_for(path)(path, spec_only=True)
    return ModelSpec.from_json(stored) if stored else None


def read_model_state(path: Path) -> tuple[dict[str, torch.Tensor], ModelSpec | None]:
    """Read a state dict and the spec stored beside it, `None` if there is none.

    Args:
        path: File to read.
    """
    state, stored = reader_for(path)(path)
    return state, ModelSpec.from_json(stored) if stored else None


def write_model_state(
    path: Path, state: dict[str, torch.Tensor], spec: ModelSpec | None = None
) -> None:
    """Write a state dict with the spec that rebuilds its model, if any.

    Args:
        path: File to write.
        state: Tensors to store.
        spec: How to rebuild the model, stored beside the tensors.

    Raises:
        ValueError: If the suffix names no known format.
    """
    write_state(path, state, spec.to_json() if spec is not None else None)


def load_model(
    path: Path, strict: bool = True, trusted: bool = False, **overrides: Any
) -> torch.nn.Module:
    """Rebuild a model from a file and load its weights.

    Args:
        path: File to read.
        strict: Whether every key must match, as for `load_state_dict`.
        trusted: Let the spec name anything, for a file you trust.
        overrides: Constructor arguments to replace before building. Changing
            one that affects a tensor's shape makes the weights unloadable,
            which `strict` does not cover.

    Raises:
        ValueError: If the file carries no spec, since the architecture cannot
            be inferred from weights alone.
    """
    state, spec = read_model_state(path)
    if spec is None:
        raise ValueError(
            f"No model spec in {str(path)!r}; weights alone do not say which "
            "architecture built them. Save it with write_model_state(..., spec=...)."
        )
    model = spec.build(trusted=trusted, **overrides)
    model.load_state_dict(state, strict=strict)
    return model

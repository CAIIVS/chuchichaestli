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
from typing import Any
import torch


__all__ = ["ModelSpec", "InitArgMixin"]

_INIT_ARGS = "_init_args"
_TAG = "__c3li__"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What it takes to rebuild a model.

    Attributes:
        cls: Importable name of the class, as `"module:QualName"`.
        kwargs: Every constructor argument, defaults included.
    """

    cls: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

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
        module = getattr(obj, "__module__", None)
        for attribute in ("__qualname__", "__name__"):
            name = getattr(obj, attribute, None)
            if module is None or name is None:
                continue
            path = f"{module}:{name}"
            try:
                if ModelSpec.import_class(path) is obj:
                    return path
            except (ImportError, AttributeError):
                continue
        raise TypeError(
            f"Cannot record {obj!r} in a spec: no import name resolves to it."
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
        if isinstance(value, InitArgMixin):
            return value.spec.to_dict()
        if isinstance(value, torch.nn.Module):
            raise TypeError(
                f"Cannot record a {type(value).__name__} argument in a spec: it "
                "does not inherit InitArgMixin, so nothing says how to rebuild it."
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
    def decode_arg(cls, value: Any) -> Any:
        """Rebuild whatever `encode_arg` reduced, nested specs included.

        Args:
            value: Encoded argument to restore.
        """
        if isinstance(value, Mapping):
            if _TAG in value:
                return cls._decode_tagged(value)
            if set(value) == {"cls", "kwargs"}:
                return cls.from_dict(value).build()
            return {k: cls.decode_arg(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls.decode_arg(v) for v in value]
        return value

    @classmethod
    def _decode_tagged(cls, value: Mapping[str, Any]) -> Any:
        """Rebuild an argument `encode_arg` tagged with its kind.

        Args:
            value: Tagged mapping to restore.

        Raises:
            ValueError: If the tag names no known kind.
        """
        match value[_TAG]:
            case "import":
                return cls.import_class(value["name"])
            case "partial":
                return functools.partial(
                    cls.decode_arg(value["func"]),
                    *cls.decode_arg(value["args"]),
                    **cls.decode_arg(value["kwargs"]),
                )
            case "tensor":
                dtype = getattr(torch, value["dtype"])
                if dtype.is_complex:
                    parts = torch.tensor(value["data"], dtype=torch.float64)
                    parts = parts.reshape(*value["shape"], 2)
                    return torch.view_as_complex(parts).to(dtype)
                return torch.tensor(value["data"], dtype=dtype).reshape(value["shape"])
            case "dtype":
                return getattr(torch, value["name"])
            case "device":
                return torch.device(value["name"])
            case "enum":
                return cls.import_class(value["cls"])(cls.decode_arg(value["value"]))
            case "path":
                return Path(value["path"])
            case "tuple":
                return tuple(cls.decode_arg(v) for v in value["items"])
            case "mapping":
                return {cls.decode_arg(k): cls.decode_arg(v) for k, v in value["items"]}
        raise ValueError(f"Unknown encoded argument kind {value[_TAG]!r}.")

    def build(self, **overrides: Any) -> Any:
        """Construct the object this spec describes.

        Args:
            overrides: Arguments to replace, for rebuilding a variant.
        """
        kwargs = {key: self.decode_arg(value) for key, value in self.kwargs.items()}
        return self.import_class(self.cls)(**{**kwargs, **overrides})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the spec."""
        kwargs = {key: self.encode_arg(value) for key, value in self.kwargs.items()}
        return {"cls": self.cls, "kwargs": kwargs}

    def to_json(self) -> str:
        """Return the spec as one JSON string, for file metadata."""
        return json.dumps(self.to_dict(), sort_keys=True)

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
        """
        return cls.from_dict(json.loads(text))


class InitArgMixin:
    """Records every constructor argument, so `.spec` can rebuild the object."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap a subclass's `__init__` so its arguments are recorded.

        Args:
            kwargs: Passed to the next `__init_subclass__` in the chain.
        """
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if not inspect.isfunction(init) or getattr(init, "_records_init", False):
            return
        signature = inspect.signature(init)

        @functools.wraps(init)
        def recorded(self: Any, *args: Any, **kwargs: Any) -> None:
            """Run the original `__init__`, remembering what it was given.

            Args:
                self: The object being constructed.
                args: Positional arguments, as the caller gave them.
                kwargs: Keyword arguments, as the caller gave them.
            """
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            recorded_args = dict(bound.arguments)
            recorded_args.pop("self", None)
            for name, parameter in signature.parameters.items():
                if parameter.kind is parameter.VAR_KEYWORD:
                    recorded_args.update(recorded_args.pop(name, {}))
                elif parameter.kind is parameter.VAR_POSITIONAL:
                    recorded_args[name] = list(recorded_args.get(name, ()))
            init(self, *args, **kwargs)
            object.__setattr__(self, _INIT_ARGS, recorded_args)

        recorded._records_init = True
        cls.__init__ = recorded

    @property
    def spec(self) -> ModelSpec:
        """Return what it takes to rebuild this object.

        Raises:
            AttributeError: If the arguments were never recorded, which means
                the class defines no `__init__` of its own to wrap.
        """
        recorded = getattr(self, _INIT_ARGS, None)
        if recorded is None:
            raise AttributeError(
                f"{type(self).__name__} recorded no constructor arguments; "
                "it defines no __init__ for InitArgMixin to wrap."
            )
        return ModelSpec(cls=ModelSpec.qualname(type(self)), kwargs=dict(recorded))

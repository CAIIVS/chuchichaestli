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

    @classmethod
    def encode_arg(cls, value: Any) -> Any:
        """Reduce a constructor argument to something JSON can hold.

        Args:
            value: Argument to reduce.

        Raises:
            TypeError: If `value` is a module that records no arguments, since
                the rebuilt model would silently differ from the saved one.
        """
        if isinstance(value, InitArgMixin):
            return value.spec.to_dict()
        if isinstance(value, torch.nn.Module):
            raise TypeError(
                f"Cannot record a {type(value).__name__} argument in a spec: it "
                "does not inherit InitArgMixin, so nothing says how to rebuild it."
            )
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(k): cls.encode_arg(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls.encode_arg(v) for v in value]
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        if isinstance(value, type):
            return cls.qualname(value)
        return repr(value)

    @classmethod
    def decode_arg(cls, value: Any) -> Any:
        """Rebuild whatever `encode_arg` reduced, nested specs included.

        Args:
            value: Encoded argument to restore.
        """
        if isinstance(value, Mapping):
            if set(value) == {"cls", "kwargs"}:
                return cls.from_dict(value).build()
            return {k: cls.decode_arg(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls.decode_arg(v) for v in value]
        return value

    def build(self, **overrides: Any) -> Any:
        """Construct the object this spec describes.

        Args:
            overrides: Arguments to replace, for rebuilding a variant.
        """
        kwargs = {key: self.decode_arg(value) for key, value in self.kwargs.items()}
        return self.import_class(self.cls)(**{**kwargs, **overrides})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the spec."""
        return {"cls": self.cls, "kwargs": self.encode_arg(dict(self.kwargs))}

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

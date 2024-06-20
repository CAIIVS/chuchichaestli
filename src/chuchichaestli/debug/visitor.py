"""Visitor pattern implementation for PyTorch models.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

from dataclasses import dataclass
from enum import Enum

from typing import Any

from torch import nn


class HookDirection(Enum):
    """Enum that represents the direction of a hook."""

    Forward: str = "forward"
    Backward: str = "backward"
    Both: str = "both"


@dataclass
class Hook:
    """Represents a hook (function to call and direction to attach the hook)."""

    fn: callable[[nn.Module, ...], Any]
    direction: HookDirection


class Visitor:
    """Generic implementation of a Visitor."""

    def __init__(
        self,
        hook_default: Hook,
        hook_map: dict[object, Hook] = {},
        max_depth: int | None = None,
    ):
        """Initialize the Visitor.

        Args:
            hook_default: The default hook to use.
            hook_map: A mapping from layer types to hooks.
            max_depth: The maximum depth to visit.
        """
        self.hook_default = hook_default
        self.hook_map = hook_map
        self.max_depth = max_depth

    def visit(self, visitee, depth: int = 0, caller: nn.Module | None = None):
        """Function called when visiting an object."""
        if depth > self.max_depth:
            return
        for _, layer in visitee.modules():
            if caller is layer:
                continue
            layer_type = type(layer)
            hook = self.hook_map.setdefault(layer_type, self.hook_default)
            match hook.direction:
                case HookDirection.Forward:
                    layer.register_forward_hook(hook.fn, layer, depth)
                case HookDirection.Backward:
                    layer.register_backward_hook(hook.fn, layer, depth)
                case HookDirection.Both:
                    layer.register_forward_hook(hook.fn, layer, depth)
                    layer.register_backward_hook(hook.fn, layer, depth)
            self.visit(layer, depth=depth + 1, caller=layer)

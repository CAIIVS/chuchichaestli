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

from collections.abc import Callable
from typing import Any

from torch import nn
from chuchichaestli.debug import as_bytes, cli_pbar


class HookDirection(Enum):
    """Enum that represents the direction of a hook."""

    Forward: str = "forward"
    Backward: str = "backward"
    Both: str = "both"


@dataclass
class Hook:
    """Represents a hook (function to call and direction to attach the hook)."""

    fn: Callable[[nn.Module, Any], Any]
    direction: HookDirection


class Visitor:
    """Generic implementation of a Visitor."""

    def __init__(
        self,
        hook_default: Hook | None = None,
        hook_map: dict[object, Hook] = {},
        max_depth: int | None = None,
        _memory_stats: list = [],
    ):
        """Initialize the Visitor.

        Args:
            hook_default: The default hook to use.
            hook_map: A mapping from layer types to hooks.
            max_depth: The maximum depth to visit.
            _memory_stats: Precached memory statistics history.
        """
        self.hook_default = hook_default
        self.hook_map = hook_map
        self.max_depth = max_depth
        self._memory_stats = _memory_stats

    @property
    def memory_stats(self):
        """Return the collected memory statistics."""
        return self._memory_stats

    def unlink(self):
        """Unlink hooks and maps."""
        self.hook_default = None
        self.hook_map = {}

    def visit(self, visitee, depth: int = 0, caller: nn.Module | None = None):
        """Function called when visiting an object."""
        # print(visitee, depth, caller)
        if self.max_depth and depth > self.max_depth:
            return
        for layer in visitee.modules():
            if caller is layer:
                return
            layer_type = type(layer)
            hook = self.hook_map.setdefault(layer_type, self.hook_default)
            match hook.direction:
                case HookDirection.Forward:
                    layer.register_forward_hook(hook.fn)
                case HookDirection.Backward:
                    layer.register_backward_hook(hook.fn)
                case HookDirection.Both:
                    layer.register_forward_hook(hook.fn)
                    layer.register_backward_hook(hook.fn)
            self.visit(layer, depth=depth + 1, caller=layer)

    def report(
        self,
        unit: str = "MB",
        with_bar: bool = True,
        bar_length: int = 60,
        verbose: bool = True,
    ) -> list[str] | None:
        """Report the memory statistics for the visited module(s).

        Args:
            unit (str): The byte unit; one in [KB, MB, GB, TB, KiB, MiB, GiB, TiB].
            with_bar (bool): Add simple horizontal percentage bars.
            bar_length (int): The length of the percentage bars in number of characters.
            verbose (bool): Directly print lines to stdout.
        """
        if not hasattr(self, "memory_stats"):
            return []
        divisor = as_bytes(unit)
        mem_stats = self.memory_stats.copy()
        if not mem_stats:
            return []
        total_alloc = sum(module["allocated"] for module in mem_stats)
        for stat in mem_stats:
            stat["rel_alloc"] = stat["allocated"] / total_alloc
            stat["allocated"] /= divisor
        total_alloc /= divisor

        lines = []
        module_name_length = max([len(stat["module"]) for stat in mem_stats])

        for stat in mem_stats:
            prefix = [stat["module"].ljust(module_name_length), stat["allocated"], unit]
            postfix = [stat["rel_alloc"] * 100, "%"]
            line = cli_pbar(stat["rel_alloc"], prefix, postfix, bar_length=bar_length)
            lines.append(line)
        if verbose:
            print("\n".join(lines))
        return lines

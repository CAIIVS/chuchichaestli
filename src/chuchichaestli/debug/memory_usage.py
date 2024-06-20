"""Memory debugging utils.

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

import torch
from torch import nn
from chuchichaestli.debug.visitor import Visitor, Hook, HookDirection


class MemoryStatsVisitor(Visitor):
    """Visitor that collects memory statistics."""

    def __init__(self):
        """Initialize the visitor."""
        super().__init__()
        self._memory_stats = []
        self._hook_default = Hook(self._hook, HookDirection.Forward)

    def _hook(self, module: nn.Module):
        """Hook that collects the memory statistics."""
        mem_dict = torch.cuda.memory_stats()
        mem_dict["module"] = module._get_name()
        self._memory_stats.append(torch.cuda.memory_stats())

    @property
    def memory_stats(self):
        """Return the collected memory statistics."""
        return self._memory_stats

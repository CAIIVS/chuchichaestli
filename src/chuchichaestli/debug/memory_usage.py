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


class CudaMemoryStatsVisitor(Visitor):
    """Visitor that collects memory statistics on CUDA GPUs."""

    def __init__(self, **kwargs):
        """Initialize the visitor."""
        if "hook_default" in kwargs:
            hook_default = kwargs.pop("hook_default")
        else:
            hook_default = Hook(self._hook, HookDirection.Forward)
        super().__init__(hook_default=hook_default, **kwargs)

    def _hook(self, module: nn.Module, _input: torch.Tensor, _output: torch.Tensor):
        """Hook that collects the memory statistics."""
        mem_dict = torch.cuda.memory_stats()
        mem_dict["module"] = module._get_name()
        self._memory_stats.append(mem_dict)

    @staticmethod
    def has_cuda(verbose: bool = False) -> bool:
        """Check if CUDA is built and available."""
        is_avail = torch.cuda.is_available()
        is_built = torch.backends.cuda.is_built()
        is_cuda = is_avail and is_built
        if verbose:
            if not is_built:
                if not is_avail:
                    print(
                        "CUDA not available most likely because you do not have a CUDA-enabled "
                        "device on this machine."
                    )
                else:
                    print(
                        "CUDA not available because the current PyTorch install was not built with "
                        "CUDA enabled."
                    )
        return is_cuda


class MPSMemoryAllocationVisitor(Visitor):
    """Visitor that collects memory statistics on Apple systems."""

    def __init__(self, **kwargs):
        """Initialize the visitor."""
        if "hook_default" in kwargs:
            hook_default = kwargs.pop("hook_default")
        else:
            hook_default = Hook(self._hook, HookDirection.Forward)
        super().__init__(hook_default=hook_default, **kwargs)

    def _hook(self, module: nn.Module, _input: torch.Tensor, _output: torch.Tensor):
        """Hook that collects the memory statistics."""
        allocated = torch.mps.current_allocated_memory()
        mem_dict = {"allocated": allocated, "module": module._get_name()}
        self._memory_stats.append(mem_dict)

    @staticmethod
    def has_mps(verbose: bool = False) -> bool:
        """Check if MPS is built and available."""
        is_avail = torch.backends.mps.is_available()
        is_built = torch.backends.mps.is_built()
        is_mps = is_avail and is_built
        if verbose:
            if not is_avail:
                if not is_built:
                    print(
                        "MPS not available because the current PyTorch install was not built with "
                        "MPS enabled."
                    )
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ and/or "
                        "you do not have an MPS-enabled device on this machine."
                    )
        return is_mps

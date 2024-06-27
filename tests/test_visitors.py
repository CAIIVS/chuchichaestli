"""Tests for visitors.

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
from chuchichaestli.debug.memory_usage import (
    CudaMemoryStatsVisitor,
    MPSMemoryAllocationVisitor,
)
from chuchichaestli.models.resnet import ResidualBlock


def test_cuda_memory_stats_visitor():
    """Test the CUDA memory stats visitor."""
    visitor = CudaMemoryStatsVisitor()
    res_block = ResidualBlock(3, 64, 32, True, 64)
    visitor.visit(res_block)
    res_block(torch.randn(1, 64, 32, 32, 32), torch.randn(1, 64))
    assert (
        len(visitor.memory_stats) == 10 + 2 + 1
    )  # 10 layers + 2 for the Norm internals + resnet block itself


def _test_mps_memory_allocation_visitor():
    """Test the MPS memory allocation visitor."""
    visitor = MPSMemoryAllocationVisitor()
    res_block = ResidualBlock(3, 64, 32, True, 64)
    visitor.visit(res_block)
    res_block(torch.randn(1, 64, 32, 32, 32), torch.randn(1, 64))
    assert len(visitor.memory_stats) == 13 + 1  # 10 layers + resnet block itself

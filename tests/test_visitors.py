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


import pytest
import torch
from chuchichaestli.debug.memory_usage import (
    CudaMemoryStatsVisitor,
    MPSMemoryAllocationVisitor,
)
from chuchichaestli.models.resnet import ResidualBlock


def test_cuda_memory_stats_visitor():
    """Test the CUDA memory stats visitor."""
    if not CudaMemoryStatsVisitor.has_cuda():
        print("No CUDA device found!")
        return
    visitor = CudaMemoryStatsVisitor()
    res_block = ResidualBlock(3, 64, 32, True, 64)
    visitor.visit(res_block)
    res_block(torch.randn(1, 64, 32, 32, 32), torch.randn(1, 64))
    assert (
        len(visitor.memory_stats) == 10 + 2 + 1
    )  # 10 layers + 2 for the Norm internals + resnet block itself


@pytest.fixture
def mps_visitor(request):
    """Initialize and cache fixture for a stripped MPS visitor."""
    v_dict = request.config.cache.get("mps_visitor", None)
    if not MPSMemoryAllocationVisitor.has_mps():
        print("No MPS device found!")
        return v_dict
    elif v_dict is None:
        torch.mps.empty_cache()
        mps_device = torch.device("mps")
        # init visitor
        visitor = MPSMemoryAllocationVisitor()
        # Conv3D is not supported on MPS
        res_block = ResidualBlock(2, 64, 32, True, 64)
        res_block.to(mps_device)
        visitor.visit(res_block)
        res_block(
            torch.randn(1, 64, 32, 32, device=mps_device),
            torch.randn(1, 64, device=mps_device),
        )
        # remove hook references for JSON caching
        visitor.unlink()
        request.config.cache.set("mps_visitor", visitor.__dict__)
    else:
        visitor = MPSMemoryAllocationVisitor(**v_dict)
    return visitor


def test_mps_memory_allocation(mps_visitor):
    """Test the MPS memory allocation visitor."""
    visitor = mps_visitor
    if visitor:
        assert len(visitor.memory_stats) == 10 + 1  # 10 layers + resnet block itself


def test_mps_memory_report(mps_visitor):
    """Test the MPS memory allocation visitor."""
    visitor = mps_visitor
    if visitor:
        print()
        rep = visitor.report()
        assert isinstance(rep, list)
        assert len(rep) == 10 + 1
        assert isinstance(rep[0], str)

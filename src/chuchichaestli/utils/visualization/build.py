# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a semantic IR graph from a PyTorch model."""

from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from chuchichaestli.utils.modules import info_forward_pass
from chuchichaestli.utils.visualization.adapters import AdapterRegistry, default_registry
from chuchichaestli.utils.visualization.ir import IRGraph

__all__ = ["build_ir"]


def build_ir(
    model: nn.Module,
    input_shape: Sequence[int] | torch.Size | None = None,
    input_dtype: torch.dtype = torch.float32,
    registry: AdapterRegistry | None = None,
) -> IRGraph:
    """Build a semantic IR graph for a model.

    Traces the module once (with shapes if `input_shape` is given, structure-only
    otherwise), resolves an adapter, and returns the finalized graph.

    Args:
        model: Model to visualize.
        input_shape: Input shape for shape tracing; structure-only if None.
        input_dtype: Input dtype for tracing.
        registry: Adapter registry (defaults to the built-in one).
    """
    if input_shape is not None:
        info_list = info_forward_pass(
            model, input_shape=input_shape, input_dtype=input_dtype, use_cache=False
        )
    else:
        info_list = info_forward_pass(model, use_cache=False)
    info_by_id = {info.layer_id: info for info in info_list}
    adapter = (registry or default_registry()).resolve(model)
    return adapter.build(model, info_by_id).finalize()

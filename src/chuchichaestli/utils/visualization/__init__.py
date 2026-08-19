# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualization utilities for chuchichaestli models."""

from collections.abc import Sequence
import torch
from torch import nn
from chuchichaestli.utils.visualization.colors import (
    list_color_names,
    get_color,
    color_variant,
)
from chuchichaestli.utils.visualization.ir import (
    IRGraph,
    IRNode,
    IREdge,
    NodeRole,
    EdgeKind,
)
from chuchichaestli.utils.visualization.build import build_ir
from chuchichaestli.utils.visualization.base import Renderer, DiagramLevel, ZoomSpec
from chuchichaestli.utils.visualization.mermaid import MermaidDiagram, mermaid_diagram

__all__ = [
    "list_color_names",
    "get_color",
    "color_variant",
    "IRGraph",
    "IRNode",
    "IREdge",
    "NodeRole",
    "EdgeKind",
    "build_ir",
    "Renderer",
    "DiagramLevel",
    "ZoomSpec",
    "MermaidDiagram",
    "mermaid_diagram",
    "matplotlib_diagram",
]


def matplotlib_diagram(
    model: nn.Module,
    level: int | str = 0,
    zoom: ZoomSpec | str | None = None,
    input_shape: Sequence[int] | torch.Size | None = None,
    input_dtype: torch.dtype = torch.float32,
    show_params: bool = True,
    show_legend: bool = True,
    title: str | None = None,
):
    """Create a matplotlib diagram of a model at a given abstraction level.

    Args:
        model: Model instance from chuchichaestli (or another PyTorch model).
        level: Abstraction level (0=components ... 3=layers).
        zoom: Optional exemplary-zoom target (node id or `ZoomSpec`).
        input_shape: Input shape for shape tracing; structure-only if None.
        input_dtype: Input dtype for tracing.
        show_params: Whether to annotate parameter counts.
        show_legend: Whether to draw a type legend.
        title: Optional figure title.
    """
    from chuchichaestli.utils.visualization.mpl import MatplotlibRenderer

    graph = build_ir(model, input_shape=input_shape, input_dtype=input_dtype)
    return MatplotlibRenderer(
        graph,
        level=level,
        zoom=zoom,
        show_params=show_params,
        show_legend=show_legend,
        title=title,
    )

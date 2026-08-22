# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualization utilities for chuchichaestli models."""

from collections.abc import Sequence
import torch
from torch import nn
from chuchichaestli.utils.ir import (
    IRGraph,
    IRNode,
    IREdge,
    NodeRole,
    EdgeKind,
    GraphLevel,
    build_ir,
)
from chuchichaestli.utils.info import summary
from chuchichaestli.utils.visualization.colors import (
    list_color_names,
    get_color,
    color_variant,
    PALETTE,
)
from chuchichaestli.utils.visualization.base import Renderer, ZoomSpec, LabelField
from chuchichaestli.utils.visualization.mpl import DiagramStyle, ColorMode
from chuchichaestli.utils.visualization.mermaid import MermaidDiagram, DiagramDirection

__all__ = [
    "list_color_names",
    "get_color",
    "color_variant",
    "PALETTE",
    "IRGraph",
    "IRNode",
    "IREdge",
    "NodeRole",
    "EdgeKind",
    "build_ir",
    "summary",
    "Renderer",
    "GraphLevel",
    "ZoomSpec",
    "DiagramStyle",
    "MermaidDiagram",
    "mermaid_diagram",
    "matplotlib_diagram",
]


def matplotlib_diagram(
    model: nn.Module,
    level: int | str = 0,
    zoom: ZoomSpec | str | bool | list[ZoomSpec | str] | None = None,
    input_shape: Sequence[int] | torch.Size | None = None,
    input_dtype: torch.dtype = torch.float32,
    show_legend: bool = True,
    color_labels: bool = False,
    title: str | None = None,
    label_fields: Sequence[LabelField] | None = None,
    color_by: ColorMode | None = None,
    node_size: str | None = None,
    zoom_loc: str = "right",
    zoom_size: float = 0.22,
    zoom_bounds: tuple[float, float, float, float] | None = None,
    zoom_fields: Sequence[str] | None = None,
    style: DiagramStyle | None = None,
):
    """Create a matplotlib diagram of a model at a given abstraction level.

    Args:
        model: Model instance from chuchichaestli (or another PyTorch model).
        level: Abstraction level (0=components ... 3=layers).
        zoom: Exemplary-zoom target(s). A node id (str) or `ZoomSpec` zooms that
            node; `True` auto-picks a representative block (first encoder
            block); a list of ids/`ZoomSpec`s draws several insets (give each a
            distinct `loc` via `ZoomSpec`, e.g. one `"left"` and one `"right"`);
            None disables zoom.
        input_shape: Input shape for shape tracing; structure-only if None.
        input_dtype: Input dtype for tracing.
        show_legend: Whether to draw a legend.
        color_labels: Deprecated alias for `color_by="name"`.
        title: Optional figure title.
        label_fields: Which fields to show inside each node, any of `"name"`,
            `"channels"` (in->out), `"kernel"`, `"resolution"`, `"params"`. None
            uses a per-role default (channels+resolution for levels, name
            elsewhere, both +params).
        color_by: What fill colours (and the legend) encode: `"component"`,
            `"type"`, or `"name"`; None auto-selects.
        node_size: Node width/height scale: `"small"`, `"medium"`, or `"large"`.
            None defaults to `"medium"`.
        zoom_loc: Inset placement when `zoom` is a single id/`True` (see
            `ZoomSpec`); ignored for a list (use per-`ZoomSpec` `loc`).
        zoom_size: Inset size fraction when `zoom` is a single id/`True`.
        zoom_bounds: Explicit `[x, y, w, h]` inset bounds; overrides `zoom_loc`/
            `zoom_size`.
        zoom_fields: Comma-separated label fields for zoom layers (see
            `ZoomSpec`), applied when `zoom` is a single id/`True`.
        style: Layout/typography overrides; None uses `DiagramStyle()`.
    """
    from chuchichaestli.utils.visualization.mpl import MatplotlibRenderer

    graph = build_ir(model, input_shape=input_shape, input_dtype=input_dtype)
    if zoom is True:
        rep = graph.representative(NodeRole.BLOCK)
        if rep is None:
            raise ValueError("No block to zoom into; pass an explicit target.")
        zoom = rep.id
    if isinstance(zoom, str):
        fields = tuple(zoom_fields) if zoom_fields is not None else None
        zoom = ZoomSpec(
            zoom, loc=zoom_loc, size=zoom_size, bounds=zoom_bounds, fields=fields
        )
    return MatplotlibRenderer(
        graph,
        level=level,
        zoom=zoom,
        show_legend=show_legend,
        color_labels=color_labels,
        title=title,
        label_fields=label_fields,
        color_by=color_by,
        node_size=node_size,
        style=style,
    )


def mermaid_diagram(
    model: nn.Module,
    direction: DiagramDirection = "horizontal",
    max_depth: int | None = None,
    positions: dict | None = None,
    label_fields: Sequence[LabelField] = ("name", "params"),
    show_legend: bool = False,
    **kwargs,
) -> MermaidDiagram:
    """Create a mermaid diagram from a model.

    Args:
        model: Model instance from chuchichaestli.
        direction: Diagram direction, e.g. T(op)D(own), L(eft)R(ight), etc.
        max_depth: Maximum depth for module recursion; None uses full depth.
        positions: Mapping of node IDs to custom positions.
        label_fields: Which fields to label nodes with, any of `"name"`,
            `"channels"`, `"resolution"`, `"params"` (`"channels"`/`"resolution"`
            both show tensor shapes; `"kernel"` is not rendered by this backend).
        show_legend: Whether to show the legend explaining components.
        kwargs: Other keyword arguments for `MermaidDiagram`.

    Note:
        The diagram generation might work on other PyTorch modules, but is not
        guaranteed.
    """
    return MermaidDiagram(
        model,
        direction=direction,
        positions=positions,
        max_depth=max_depth,
        label_fields=label_fields,
        show_legend=show_legend,
        **kwargs,
    )

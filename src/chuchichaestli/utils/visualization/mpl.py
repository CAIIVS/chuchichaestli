# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Matplotlib backend for publication-quality model diagrams."""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, get_args
from chuchichaestli.utils.formatting import metric_suffix
from chuchichaestli.utils.visualization.base import (
    Renderer,
    ZoomSpec,
    LabelField,
    LABEL_FIELDS,
    type_color,
    type_fill,
    component_color,
)
from chuchichaestli.utils.visualization.colors import get_color, color_variant, PALETTE
from chuchichaestli.utils.ir import (
    IRGraph,
    IRNode,
    EdgeKind,
    NodeRole,
    normalize_level,
)

__all__ = ["DiagramStyle", "MatplotlibRenderer"]


@dataclass(frozen=True)
class DiagramStyle:
    """Layout and typography constants for `MatplotlibRenderer`.

    Lengths are in data units; `node_size` scales the box/spacing lengths by
    the matching `node_size_scale` multiplier.

    Args:
        h_min: Minimum node half-height.
        h_max: Maximum node half-height.
        h_mid: Half-height for nodes without a resolution size hint.
        res_bulge: Residual-shortcut arc depth below a block's baseline.
        slot: Per-node horizontal slot width.
        gap: Horizontal gap between slots.
        ipu: Inches per data unit (deterministic layout scale).
        fs: Base label font size (points).
        max_figw: Figure width ceiling (inches); font shrinks to fit if exceeded.
        node_size_scale: `node_size` keyword to length-multiplier map.
    """

    h_min: float = 1.4
    h_max: float = 4.6
    h_mid: float = 2.6
    res_bulge: float = 0.9
    slot: float = 1.6
    gap: float = 0.9
    ipu: float = 0.85
    fs: float = 10.5
    max_figw: float = 200.0
    node_size_scale: Mapping[str, float] = field(
        default_factory=lambda: {"small": 0.72, "medium": 1.0, "large": 1.4}
    )


ColorMode = Literal["auto", "component", "type", "name"]
_LABEL_FIELDS = frozenset(LABEL_FIELDS)
_COLOR_MODES = frozenset(get_args(ColorMode))
# Comma-separated fields shown on zoom-inset layers (wide-but-short boxes).
_ZOOM_FIELDS = LABEL_FIELDS


def _require_mpl() -> SimpleNamespace:
    """Import matplotlib lazily with a helpful error if it is missing."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import (
            Polygon,
            FancyBboxPatch,
            FancyArrowPatch,
            ConnectionPatch,
            Patch,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Matplotlib is required for this backend. "
            "Install it with: pip install 'chuchichaestli[viz]'"
        ) from exc
    return SimpleNamespace(
        matplotlib=matplotlib,
        plt=plt,
        Polygon=Polygon,
        FancyBboxPatch=FancyBboxPatch,
        FancyArrowPatch=FancyArrowPatch,
        ConnectionPatch=ConnectionPatch,
        Patch=Patch,
    )


def _side(node_id: str) -> str:
    """Top-level component key from a node id."""
    parts = node_id.split("/")
    return parts[1] if len(parts) > 1 else "model"


def _dock_anchors(loc: str) -> tuple[str | None, str | None]:
    """Parse a zoom loc into (horizontal, vertical) anchors.

    `horizontal` is `"left"`/`"right"`/None, `vertical` is
    `"top"`/`"bottom"`/None. A side edge (one anchor) docks along that edge;
    a corner (both anchors) hangs off that corner; neither is centred.
    """
    horz = "left" if "left" in loc else "right" if "right" in loc else None
    vert = (
        "top"
        if ("top" in loc or "above" in loc)
        else "bottom"
        if ("bottom" in loc or "below" in loc)
        else None
    )
    return horz, vert


class MatplotlibRenderer(Renderer):
    """Render an `IRGraph` as trapezoid/box schematics via matplotlib."""

    def __init__(
        self,
        graph: IRGraph,
        level: int | str = 0,
        zoom: ZoomSpec | str | list[ZoomSpec | str] | None = None,
        show_legend: bool = True,
        color_labels: bool = False,
        title: str | None = None,
        label_fields: Sequence[LabelField] | None = None,
        color_by: ColorMode | None = None,
        node_size: str | None = None,
        style: DiagramStyle | None = None,
    ) -> None:
        """Constructor.

        Args:
            graph: Finalized semantic IR graph.
            level: Abstraction level (0=components ... 3=layers).
            zoom: Optional exemplary-zoom target(s). A node id or `ZoomSpec`, or
                a list of them to draw several insets at different locations.
            show_legend: Whether to draw a legend.
            color_labels: Deprecated alias for `color_by="name"`.
            title: Optional figure title.
            label_fields: Which fields to show inside each node, any of
                `"name"`, `"channels"` (in->out), `"kernel"`,
                `"resolution"`, `"params"`. None uses a per-role default
                (channels+resolution for levels, name elsewhere, both +params).
            color_by: What the fill colours (and legend) encode: `"component"`
                (encoder/bottleneck/decoder), `"type"` (layer/block type), or
                `"name"`. None auto-selects (component for groups, type for
                layers), or `"name"` when `color_labels` is set.
            node_size: Node width/height scale: `"small"`, `"medium"`, or
                `"large"`. None defaults to `"medium"`.
            style: Layout/typography overrides; None uses `DiagramStyle()`.
        """
        super().__init__(graph)
        self.style = style or DiagramStyle()
        self.depth = normalize_level(level)
        if node_size is not None and node_size not in self.style.node_size_scale:
            raise ValueError(
                f"Unknown node_size {node_size!r}; choose from "
                f"{sorted(self.style.node_size_scale)}"
            )
        s = self.style.node_size_scale[node_size or "medium"]
        self._slot, self._gap = self.style.slot * s, self.style.gap * s
        self._h_min, self._h_max, self._h_mid = (
            self.style.h_min * s,
            self.style.h_max * s,
            self.style.h_mid * s,
        )
        self._res_bulge = self.style.res_bulge * s
        self._fs = self.style.fs  # recomputed in render() if the width is clamped
        items = zoom if isinstance(zoom, list | tuple) else [zoom]
        self.zooms: list[ZoomSpec] = []
        for z in items:
            if z is None:
                continue
            if isinstance(z, str):
                z = ZoomSpec(z)
            if z.target not in graph.index:
                raise graph._unknown_zoom_error(z.target)
            if z.fields is not None:
                bad = [f for f in z.fields if f not in _LABEL_FIELDS]
                if bad:
                    raise ValueError(
                        f"Unknown zoom field(s) {bad}; choose from "
                        f"{sorted(_LABEL_FIELDS)}"
                    )
            self.zooms.append(z)
        if label_fields is not None:
            unknown = [f for f in label_fields if f not in _LABEL_FIELDS]
            if unknown:
                raise ValueError(
                    f"Unknown label field(s) {unknown}; choose from "
                    f"{sorted(_LABEL_FIELDS)}"
                )
        if color_by is not None and color_by not in _COLOR_MODES:
            raise ValueError(
                f"Unknown color_by {color_by!r}; choose from {sorted(_COLOR_MODES)}"
            )
        self.show_legend = show_legend
        self.label_fields = tuple(label_fields) if label_fields is not None else None
        self.color_by = color_by or ("name" if color_labels else "auto")
        self.title = title
        self._sizes = self._collect_sizes()
        self._label_colors: dict[str, str] = {}
        self._spanned: set[str] = set()  # nodes a forward arrow passes over

    def render(self) -> Any:
        """Build and return the matplotlib `Figure`."""
        mpl = _require_mpl()
        # Plain view (not zoom) so skip/residual edges stay on drawn boxes.
        view = self.graph.view(self.depth)
        # Draw the collapsed view's frontier (nodes with no children in view).
        drawables = [n for n in view.root.walk() if not n.children]
        if self.color_by == "name":
            names: list[str] = []
            for node in drawables:
                if node.label not in names:
                    names.append(node.label)
            self._label_colors = {
                name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)
            }

        n_skips = sum(1 for e in view.edges if e.kind == EdgeKind.SKIP)
        peak_extra = (2.5 + max(n_skips - 1, 0)) if n_skips else 0.0
        xspan = max(len(drawables), 1) * (self._slot + self._gap)
        yspan = self._h_max + 3.0 + peak_extra
        figw_full = self.style.ipu * xspan + 1.0
        figw = min(self.style.max_figw, figw_full)
        figh = self.style.ipu * yspan + 1.0
        # Shrink font only when width was clamped, so labels stay in their boxes.
        self._fs = self.style.fs * min(1.0, figw / figw_full)
        # Labels on nodes a forward edge skips over must dodge that arrow.
        order = {n.id: i for i, n in enumerate(drawables)}
        self._spanned = set()
        for e in view.edges:
            if e.kind != EdgeKind.FORWARD or e.source_id not in order:
                continue
            if e.target_id not in order:
                continue
            lo, hi = sorted((order[e.source_id], order[e.target_id]))
            self._spanned.update(drawables[k].id for k in range(lo + 1, hi))

        fig, ax = mpl.plt.subplots(figsize=(figw, figh))
        boxes = self._layout(mpl, ax, drawables)
        max_top = self._draw_edges(mpl, ax, view, boxes)
        for zoom in self.zooms:
            self._draw_zoom(mpl, ax, boxes, zoom)
        xmax = max((b[1] for b in boxes.values()), default=self._slot)
        hmax = max((b[3] for b in boxes.values()), default=self._h_mid)
        ax.set_xlim(-0.6, xmax + 0.6)
        ax.set_ylim(-hmax / 2 - 1.5, max(hmax / 2 + 2.5, max_top + 0.8))
        self._finish(mpl, ax, drawables)
        return fig

    def _collect_sizes(self) -> dict[str, float | None]:
        sizes: dict[str, float | None] = {}

        def rec(node: IRNode) -> float | None:
            own = float(max(node.geometry.spatial)) if node.geometry.spatial else None
            # A resample block's output resolution belongs to the next level,
            # so exclude it when sizing an ancestor.
            child_vals = [
                v
                for c in node.children
                if (v := rec(c)) is not None
                and not (c.geometry.is_downsample or c.geometry.is_upsample)
            ]
            value = own if own is not None else (max(child_vals) if child_vals else None)
            sizes[node.id] = value
            return value

        rec(self.graph.root)
        return sizes

    def _height_for(self, size: float | None) -> float:
        values = [v for v in self._sizes.values() if v]
        if not values or size is None:
            return self._h_mid
        lo, hi = math.log2(min(values)), math.log2(max(values))
        if hi == lo:
            return self._h_mid
        frac = (math.log2(size) - lo) / (hi - lo)
        return self._h_min + frac * (self._h_max - self._h_min)

    def _height(self, node_id: str) -> float:
        return self._height_for(self._sizes.get(node_id))

    def _resample_heights(self, node: IRNode) -> tuple[float, float]:
        """Left/right trapezoid heights from a resampling node's in/out sizes."""
        info = node.info
        in_sp = max(info.input_size[2:]) if info and len(info.input_size) > 2 else None
        out_sp = max(info.output_size[2:]) if info and len(info.output_size) > 2 else None
        left = self._height_for(in_sp) if in_sp else self._height(node.id)
        right = self._height_for(out_sp) if out_sp else left
        return left, right

    def _component_heights(self, node: IRNode) -> tuple[float, float]:
        source = self.graph.index.get(node.id, node)
        descendants = [
            d
            for d in source.walk()
            if d is not source
            and d.role in (NodeRole.BLOCK, NodeRole.LAYER)
            and self._sizes.get(d.id) is not None
        ]
        if not descendants:
            return self._h_mid, self._h_mid
        return self._height(descendants[0].id), self._height(descendants[-1].id)

    def _component_of(self, node: IRNode) -> IRNode | None:
        """The enclosing component node (encoder/bottleneck/decoder) of a level."""
        parts = node.id.split("/")
        if len(parts) < 3:
            return None
        return self.graph.index.get("/".join(parts[:2]))

    def _node_channels(self, node: IRNode) -> str | None:
        """The node's `in->out` channel transition (mirror-symmetric).

        For a group node (component/level) the `in` comes from its first
        block and the `out` from its last; for a leaf block/layer the node's
        own shapes are used.
        """
        source = self.graph.index.get(node.id, node)
        blocks = [
            d for d in source.walk() if d.role == NodeRole.BLOCK and d.geometry.channels
        ]
        head = blocks[0] if blocks else node
        tail = blocks[-1] if blocks else node
        info = head.info
        in_ch = info.input_size[1] if info and len(info.input_size) > 1 else None
        out_ch = tail.geometry.channels
        if not out_ch:
            return None
        return f"{in_ch}→{out_ch} ch" if in_ch and in_ch != out_ch else f"{out_ch} ch"

    def _node_resolution(self, node: IRNode) -> str | None:
        """The node's operating spatial resolution (principal, non-resampling)."""
        source = self.graph.index.get(node.id, node)
        principal = next(
            (
                d
                for d in source.walk()
                if d.role == NodeRole.BLOCK
                and d.geometry.channels
                and not (d.geometry.is_downsample or d.geometry.is_upsample)
            ),
            None,
        )
        geom = principal.geometry if principal is not None else node.geometry
        return "×".join(str(s) for s in geom.spatial) if geom.spatial else None

    def _field(self, node: IRNode, field: str) -> str | None:
        if field == "name":
            return node.label
        if field == "channels":
            return self._node_channels(node)
        if field == "resolution":
            return self._node_resolution(node)
        if field == "params":
            return metric_suffix(node.num_params, 1) if node.num_params else None
        if field == "kernel":
            k = node.info.kernel_size if node.info is not None else None
            if not k:
                return None
            dims = k if isinstance(k, list | tuple) else [k]
            return "k" + "×".join(str(x) for x in dims)
        return None

    def _fields_for(self, node: IRNode) -> tuple[str, ...]:
        if self.label_fields is not None:
            return self.label_fields
        # Per-role default: levels read as in->out / resolution; others by name.
        base = ("channels", "resolution") if node.role == NodeRole.LEVEL else ("name",)
        return (*base, "params")

    def _label(self, node: IRNode, include_name: bool = True) -> str:
        parts = []
        for node_field in self._fields_for(node):
            if node_field == "name" and not include_name:
                continue
            value = self._field(node, node_field)
            if value:
                parts.append(value)
        if parts:
            return "\n".join(parts)
        # Fall back to the name only when it wasn't deliberately omitted.
        return node.label if include_name else ""

    def _inner(self, node: IRNode) -> str:
        # In name-colour mode the name moves to the legend, so drop it inside.
        return self._label(node, include_name=self.color_by != "name")

    def _color_name(self, node: IRNode) -> str | None:
        if self.color_by == "name":
            return self._label_colors.get(node.label, "grey")
        if self.color_by == "type":
            return None
        # "component" and "auto" both colour by the enclosing component;
        # "auto" additionally leaves layers to their type colour.
        if node.role == NodeRole.COMPONENT:
            return component_color(node)
        if self.color_by == "auto" and node.role == NodeRole.LAYER:
            return None
        comp = self._component_of(node)
        return component_color(comp) if comp is not None else None

    def _fill(self, node: IRNode) -> str:
        name = self._color_name(node)
        return color_variant(name, 150) if name else type_fill(node.type_label)

    def _stroke(self, node: IRNode) -> str:
        name = self._color_name(node)
        return get_color(name) if name else type_color(node.type_label)

    def _layout(
        self, mpl: SimpleNamespace, ax: Any, drawables: list[IRNode]
    ) -> dict[str, tuple[float, float, float, float]]:
        boxes: dict[str, tuple[float, float, float, float]] = {}
        x = 0.0
        for node in drawables:
            side = _side(node.id)
            if node.role == NodeRole.COMPONENT and side in ("encoder", "decoder"):
                h_first, h_last = self._component_heights(node)
                self._trapezoid(mpl, ax, x, h_first, h_last, node)
                h = max(h_first, h_last)
            elif side == "latent" or node.meta.get("latent_kind"):
                h = self._height(node.id)
                self._hourglass(mpl, ax, x, h, node)
            elif node.geometry.is_downsample or node.geometry.is_upsample:
                h_left, h_right = self._resample_heights(node)
                self._trapezoid(mpl, ax, x, h_left, h_right, node)
                h = max(h_left, h_right)
            else:
                h = self._height(node.id)
                self._box(mpl, ax, x, h, node)
            boxes[node.id] = (x, x + self._slot, 0.0, h)
            x += self._slot + self._gap
        return boxes

    def _box(self, mpl: SimpleNamespace, ax: Any, x: float, h: float, node: IRNode) -> None:
        patch = mpl.FancyBboxPatch(
            (x, -h / 2),
            self._slot,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6,
            facecolor=self._fill(node),
            edgecolor=self._stroke(node),
        )
        ax.add_patch(patch)
        self._text(ax, x + self._slot / 2, 0, self._inner(node))

    def _trapezoid(
        self,
        mpl: SimpleNamespace,
        ax: Any,
        x: float,
        h_left: float,
        h_right: float,
        node: IRNode,
    ) -> None:
        verts = [
            (x, -h_left / 2),
            (x, h_left / 2),
            (x + self._slot, h_right / 2),
            (x + self._slot, -h_right / 2),
        ]
        ax.add_patch(
            mpl.Polygon(
                verts,
                closed=True,
                facecolor=self._fill(node),
                edgecolor=self._stroke(node),
                linewidth=1.8,
            )
        )
        self._text(ax, x + self._slot / 2, 0, self._inner(node))

    def _hourglass(self, mpl: SimpleNamespace, ax: Any, x: float, h: float, node: IRNode) -> None:
        waist = h * 0.55
        verts = [
            (x, -h / 2),
            (x, h / 2),
            (x + self._slot / 2, waist / 2),
            (x + self._slot, h / 2),
            (x + self._slot, -h / 2),
            (x + self._slot / 2, -waist / 2),
        ]
        ax.add_patch(
            mpl.Polygon(
                verts,
                closed=True,
                facecolor=self._fill(node),
                edgecolor=self._stroke(node),
                linewidth=1.8,
            )
        )
        # Lift the label off the waist only when a forward arrow passes over
        # this node (else it would cross the label at y=0).
        y = h * 0.1 if node.id in self._spanned else 0.0
        self._text(ax, x + self._slot / 2, y, self._inner(node))

    def _text(self, ax: Any, x: float, y: float, text: str) -> None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=self._fs,
            zorder=5,
            color=get_color("dark"),
        )

    def _draw_edges(
        self,
        mpl: SimpleNamespace,
        ax: Any,
        view: IRGraph,
        boxes: dict[str, tuple[float, float, float, float]],
    ) -> float:
        skips: list[tuple[tuple, tuple, str | None]] = []
        for edge in view.edges:
            # Group endpoints have no box at layer level;
            # resolve them onto the drawn frontier.
            source_id = view.on_frontier(edge.source_id, boxes, last=True)
            target_id = view.on_frontier(edge.target_id, boxes, last=False)
            if source_id is None or target_id is None or source_id == target_id:
                continue
            src, tgt = boxes[source_id], boxes[target_id]
            if edge.kind == EdgeKind.FORWARD:
                self._arrow(mpl, ax, (src[1], 0), (tgt[0], 0), "-|>", "solid", 1.4)
            elif edge.kind == EdgeKind.SKIP:
                count = edge.meta.get("count", 1)
                label = edge.label
                if count > 1:
                    label = f"{label} ×{count}" if label else f"×{count}"
                skips.append((src, tgt, label))
            elif edge.kind == EdgeKind.RESIDUAL:
                # Bend downward under the block (positive rad) with a fixed
                # shallow bulge, so long spans don't rise into the labels.
                dist = max(tgt[1] - src[0], 1e-6)
                rad = 2.0 * self._res_bulge / dist
                self._arrow(
                    mpl, ax, (src[0], -src[3] / 2), (tgt[1], -tgt[3] / 2), "-|>",
                    "dashed", 1.1, rad=rad,
                )
        max_top = self._h_mid
        for i, (src, tgt, label) in enumerate(skips):
            max_top = max(max_top, self._arc(mpl, ax, src, tgt, label, 1.0 + 0.9 * i))
        for node in view.root.walk():
            if (
                node.meta.get("has_residual")
                and node.role == NodeRole.BLOCK
                and node.id in boxes
            ):
                b = boxes[node.id]
                self._arrow(
                    mpl, ax, (b[0], -b[3] / 2), (b[1], -b[3] / 2), "-|>",
                    "dashed", 1.1, rad=-0.28,
                )
        return max_top

    def _arrow(
        self, mpl, ax, p0, p1, arrowstyle, ls, lw, rad=0.0
    ) -> None:
        ax.add_patch(
            mpl.FancyArrowPatch(
                p0,
                p1,
                arrowstyle=arrowstyle,
                mutation_scale=12,
                linewidth=lw,
                linestyle=ls,
                color=get_color("darkish"),
                connectionstyle=f"arc3,rad={rad}",
                zorder=4,
            )
        )

    def _arc(self, mpl, ax, src, tgt, label, bulge) -> float:
        """Draw a skip arc bulging `bulge` above its chord; return the apex y."""
        x0, x1 = (src[0] + src[1]) / 2, (tgt[0] + tgt[1]) / 2
        y0, y1 = src[3] / 2, tgt[3] / 2
        dist = max(abs(x1 - x0), 1e-6)
        # arc3 bulges by 0.5 * rad * dist at the apex
        rad = 2.0 * bulge / dist
        rad = -rad if x1 > x0 else rad
        ax.add_patch(
            mpl.FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.6,
                linestyle="dashed",
                color=get_color("pink"),
                connectionstyle=f"arc3,rad={rad}",
                zorder=3,
            )
        )
        apex = (y0 + y1) / 2 + bulge
        if label:
            ax.text(
                (x0 + x1) / 2, apex + 0.1, label, ha="center", va="bottom",
                fontsize=self._fs - 1, color=get_color("pink"), zorder=5,
            )
        return apex + 0.6

    def _draw_zoom(
        self,
        mpl: SimpleNamespace,
        ax: Any,
        boxes: dict[str, tuple[float, float, float, float]],
        zoom: ZoomSpec,
    ) -> None:
        target = self.graph.node(zoom.target)
        layers = [n for n in target.walk() if n.role == NodeRole.LAYER]
        if not layers:
            return
        anchor_id = "/".join(zoom.target.split("/")[: self.depth + 1])
        anchor = boxes.get(anchor_id)
        # Full-width top/bottom edge strips run the layers left-to-right; side
        # and corner columns (any left/right anchor) stack them vertically.
        horz, vert = _dock_anchors(zoom.loc)
        horizontal = vert is not None and horz is None
        inset = ax.inset_axes(self._inset_bounds(zoom))
        n = len(layers)
        inset.set_xlim(0, n if horizontal else 1)
        inset.set_ylim(0, 1 if horizontal else n)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor(get_color("gray"))
        fields = zoom.fields or _ZOOM_FIELDS
        seq = layers if horizontal else list(reversed(layers))
        for i, layer in enumerate(seq):
            label = ", ".join(v for f in fields if (v := self._field(layer, f)))
            if horizontal:
                c = i + 0.5
                inset.add_patch(
                    mpl.FancyBboxPatch(
                        (c - 0.36, 0.12),
                        0.72,
                        0.76,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=type_fill(layer.type_label),
                        edgecolor=type_color(layer.type_label),
                        linewidth=1.3,
                    )
                )
                inset.text(c, 0.5, label, ha="center", va="center",
                           fontsize=self._fs - 2, color=get_color("dark"))
                if i:
                    inset.annotate(
                        "", xy=(c - 0.36, 0.5), xytext=(c - 0.64, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=get_color("darkish")),
                    )
            else:
                c = i + 0.5
                inset.add_patch(
                    mpl.FancyBboxPatch(
                        (0.12, c - 0.36),
                        0.76,
                        0.72,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=type_fill(layer.type_label),
                        edgecolor=type_color(layer.type_label),
                        linewidth=1.3,
                    )
                )
                inset.text(0.5, c, label, ha="center", va="center",
                           fontsize=self._fs - 2, color=get_color("dark"))
                if i:
                    inset.annotate(
                        "", xy=(0.5, c - 0.36), xytext=(0.5, c - 0.64),
                        arrowprops=dict(arrowstyle="-|>", color=get_color("darkish")),
                    )
        inset.set_title(target.label, fontsize=self._fs - 1, color=get_color("dark"))
        if anchor is not None:
            b = anchor
            ax.add_patch(
                mpl.plt.Rectangle(
                    (b[0], -b[3] / 2), b[1] - b[0], b[3], fill=False,
                    edgecolor=get_color("golden"), linewidth=1.4, linestyle="dotted",
                    zorder=6,
                )
            )
            for corner in (b[3] / 2, -b[3] / 2):
                ax.add_patch(
                    mpl.ConnectionPatch(
                        xyA=((b[0] + b[1]) / 2, corner),
                        coordsA=ax.transData,
                        xyB=(0.5, 0.5),
                        coordsB=inset.transAxes,
                        color=get_color("gray"),
                        linewidth=0.8,
                        linestyle="dotted",
                        zorder=2,
                    )
                )

    def _inset_bounds(self, zoom: ZoomSpec) -> list[float]:
        if zoom.bounds is not None:
            return list(zoom.bounds)
        size = zoom.size
        horz, vert = _dock_anchors(zoom.loc)
        if horz and vert:  # corner: vertical column hanging off that corner
            col_h = 0.8
            x = 0.0 if horz == "left" else 1.0 - size
            y = 1.12 if vert == "top" else -col_h - 0.12
            return [x, y, size, col_h]
        if horz:  # side: full-height column just outside that side
            x = -size - 0.03 if horz == "left" else 1.03
            return [x, 0.0, size, 1.0]
        if vert:  # edge: full-width strip just outside that edge
            y = 1.12 if vert == "top" else -size - 0.12
            return [0.0, y, 1.0, size]
        return [0.5 - size / 2, 0.5 - size / 2, size, size]  # center

    def _finish(self, mpl: SimpleNamespace, ax: Any, drawables: list[IRNode]) -> None:
        ax.set_aspect("equal")
        ax.axis("off")
        # Keep title/legend clear of full-width top-/bottom-edge strips (corner
        # columns are left/right-aligned and don't reach the centred title/legend).
        edge = [z for z in self.zooms if _dock_anchors(z.loc)[0] is None]
        tops = [self._inset_bounds(z) for z in edge if _dock_anchors(z.loc)[1] == "top"]
        bots = [self._inset_bounds(z) for z in edge if _dock_anchors(z.loc)[1] == "bottom"]
        if self.title:
            if tops:
                y = max(b[1] + b[3] for b in tops) + 0.05
                ax.set_title(self.title, fontsize=self._fs + 3, color=get_color("dark"), y=y)
            else:
                ax.set_title(self.title, fontsize=self._fs + 3, color=get_color("dark"), pad=14)
        if self.show_legend:
            anchor_y = (min(b[1] for b in bots) - 0.06) if bots else -0.02
            self._legend(mpl, ax, drawables, anchor_y)

    def _legend_key(self, node: IRNode) -> str:
        """Legend text for a node under the active `color_by` mode."""
        if self.color_by == "name":
            return node.label
        if self.color_by == "type":
            return node.type_label
        if node.role == NodeRole.COMPONENT:
            return node.label
        if self.color_by == "auto" and node.role == NodeRole.LAYER:
            return node.type_label
        comp = self._component_of(node)
        return comp.label if comp is not None else node.type_label

    def _legend(
        self, mpl: SimpleNamespace, ax: Any, drawables: list[IRNode], anchor_y: float = -0.02
    ) -> None:
        seen: dict[str, Any] = {}
        for node in drawables:
            key = self._legend_key(node)
            if key not in seen:
                seen[key] = mpl.Patch(
                    facecolor=self._fill(node), edgecolor=self._stroke(node), label=key
                )
        if seen:
            ax.legend(
                handles=list(seen.values()),
                loc="upper center",
                bbox_to_anchor=(0.5, anchor_y),
                ncol=min(len(seen), 4),
                fontsize=self._fs - 1,
                frameon=False,
            )

    def _supported_formats(self) -> set[str]:
        """Return writable vector/raster formats."""
        return {"pdf", "svg", "png"}

    def _save_impl(
        self, artifact: Any, filepath: Path, dpi: int, aspect: float | None
    ) -> None:
        """Write the figure; `dpi` sets the resolution, `aspect` pads to a ratio.

        With no `aspect` the figure keeps its content size (tight bounding box);
        with an `aspect` the canvas is padded to that `w/h` ratio (content is
        never cropped) so the raster matches the requested shape.

        Args:
            artifact: The matplotlib figure.
            filepath: Output path.
            dpi: Target resolution.
            aspect: Target `w/h` ratio, or None for the content size.
        """
        mpl = _require_mpl()
        if aspect is not None:
            figw, figh = artifact.get_size_inches()
            if aspect > figw / figh:
                figw = figh * aspect
            else:
                figh = figw / aspect
            artifact.set_size_inches(figw, figh)
        figw, figh = artifact.get_size_inches()
        # Matplotlib rasters cannot exceed 2**16 px per side.
        dpi = min(dpi, int(65500 / max(figw, figh, 1.0)))
        artifact.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight" if aspect is None else None,
            transparent=True,
        )
        mpl.plt.close(artifact)

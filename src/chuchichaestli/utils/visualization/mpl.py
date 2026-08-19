# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Matplotlib backend for publication-quality model diagrams."""

from __future__ import annotations
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from chuchichaestli.utils.formatting import metric_suffix
from chuchichaestli.utils.visualization.base import (
    Renderer,
    ZoomSpec,
    normalize_level,
    type_color,
    type_fill,
    component_color,
)
from chuchichaestli.utils.visualization.colors import get_color, color_variant
from chuchichaestli.utils.visualization.ir import IRGraph, IRNode, EdgeKind, NodeRole

__all__ = ["MatplotlibRenderer"]

_H_MIN, _H_MAX, _H_MID = 1.4, 4.6, 2.6
_SLOT, _GAP = 2.2, 0.9
_IPU = 0.85  # inches per data unit (deterministic layout scale)
_FS = 12  # base label font size (points)
_PALETTE = [
    "blue", "green", "orange", "red", "purple", "cyan", "golden", "pink",
    "turquoise", "marguerite", "brown", "purpleblue", "yellow", "darkish",
]


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


class MatplotlibRenderer(Renderer):
    """Render an `IRGraph` as trapezoid/box schematics via matplotlib."""

    def __init__(
        self,
        graph: IRGraph,
        level: int | str = 0,
        zoom: ZoomSpec | str | None = None,
        show_params: bool = True,
        show_legend: bool = True,
        color_labels: bool = False,
        title: str | None = None,
    ) -> None:
        """Constructor.

        Args:
            graph: Finalized semantic IR graph.
            level: Abstraction level (0=components ... 3=layers).
            zoom: Optional exemplary-zoom target (node id or `ZoomSpec`).
            show_params: Whether to annotate parameter counts.
            show_legend: Whether to draw a legend.
            color_labels: If True, colour nodes by name and move names to the
                legend (keeps only the parameter count inside each shape); useful
                when block names are too long to fit.
            title: Optional figure title.
        """
        super().__init__(graph)
        self.depth = normalize_level(level)
        if isinstance(zoom, str):
            zoom = ZoomSpec(zoom)
        if zoom is not None and zoom.target not in graph.index:
            raise ValueError(f"Unknown zoom target: {zoom.target!r}")
        self.zoom = zoom
        self.show_params = show_params
        self.show_legend = show_legend
        self.color_labels = color_labels
        self.title = title
        self._sizes = self._collect_sizes()
        self._label_colors: dict[str, str] = {}

    def _collect_sizes(self) -> dict[str, float | None]:
        sizes: dict[str, float | None] = {}

        def rec(node: IRNode) -> float | None:
            own = float(max(node.geometry.spatial)) if node.geometry.spatial else None
            children = [s for s in (rec(c) for c in node.children) if s is not None]
            value = own if own is not None else (max(children) if children else None)
            sizes[node.id] = value
            return value

        rec(self.graph.root)
        return sizes

    def _height(self, node_id: str) -> float:
        values = [v for v in self._sizes.values() if v]
        size = self._sizes.get(node_id)
        if not values or size is None:
            return _H_MID
        lo, hi = math.log2(min(values)), math.log2(max(values))
        if hi == lo:
            return _H_MID
        frac = (math.log2(size) - lo) / (hi - lo)
        return _H_MIN + frac * (_H_MAX - _H_MIN)

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
            return _H_MID, _H_MID
        return self._height(descendants[0].id), self._height(descendants[-1].id)

    def _label(self, node: IRNode) -> str:
        text = node.label
        if self.show_params and node.num_params:
            text += f"\n{metric_suffix(node.num_params, 1)}"
        return text

    def _color_name(self, node: IRNode) -> str | None:
        if self.color_labels:
            return self._label_colors.get(node.label, "grey")
        if node.role == NodeRole.COMPONENT:
            return component_color(node)
        return None

    def _fill(self, node: IRNode) -> str:
        name = self._color_name(node)
        return color_variant(name, 150) if name else type_fill(node.type_label)

    def _stroke(self, node: IRNode) -> str:
        name = self._color_name(node)
        return get_color(name) if name else type_color(node.type_label)

    def _inner(self, node: IRNode) -> str:
        if self.color_labels:
            if self.show_params and node.num_params:
                return metric_suffix(node.num_params, 1)
            return ""
        return self._label(node)

    def render(self) -> Any:
        """Build and return the matplotlib `Figure`."""
        mpl = _require_mpl()
        view = (
            self.graph.zoom(self.zoom.target, self.depth)
            if self.zoom is not None
            else self.graph.view(self.depth)
        )
        drawables = [n for n in view.root.walk() if n.depth == self.depth]
        if not drawables:
            drawables = [n for n in view.root.walk() if not n.children]
        if self.color_labels:
            names: list[str] = []
            for node in drawables:
                if node.label not in names:
                    names.append(node.label)
            self._label_colors = {
                name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)
            }

        n_skips = sum(1 for e in view.edges if e.kind == EdgeKind.SKIP)
        peak_extra = (2.5 + max(n_skips - 1, 0)) if n_skips else 0.0
        xspan = max(len(drawables), 1) * (_SLOT + _GAP)
        yspan = _H_MAX + 3.0 + peak_extra
        figw = min(48.0, _IPU * xspan + 1.0)
        figh = _IPU * yspan + 1.0
        fig, ax = mpl.plt.subplots(figsize=(figw, figh))
        boxes = self._layout(mpl, ax, drawables)
        max_top = self._draw_edges(mpl, ax, view, boxes)
        if self.zoom is not None:
            self._draw_zoom(mpl, ax, boxes)
        xmax = max((b[1] for b in boxes.values()), default=_SLOT)
        hmax = max((b[3] for b in boxes.values()), default=_H_MID)
        ax.set_xlim(-0.6, xmax + 0.6)
        ax.set_ylim(-hmax / 2 - 1.5, max(hmax / 2 + 2.5, max_top + 0.8))
        self._finish(mpl, ax, drawables)
        return fig

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
                h_lo = self._height(node.id)
                h_hi = h_lo * (0.6 if node.geometry.is_downsample else 1.6)
                self._trapezoid(mpl, ax, x, h_lo, h_hi, node)
                h = max(h_lo, h_hi)
            else:
                h = self._height(node.id)
                self._box(mpl, ax, x, h, node)
            boxes[node.id] = (x, x + _SLOT, 0.0, h)
            x += _SLOT + _GAP
        return boxes

    def _box(self, mpl: SimpleNamespace, ax: Any, x: float, h: float, node: IRNode) -> None:
        patch = mpl.FancyBboxPatch(
            (x, -h / 2),
            _SLOT,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6,
            facecolor=self._fill(node),
            edgecolor=self._stroke(node),
        )
        ax.add_patch(patch)
        self._text(ax, x + _SLOT / 2, 0, self._inner(node))

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
            (x + _SLOT, h_right / 2),
            (x + _SLOT, -h_right / 2),
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
        self._text(ax, x + _SLOT / 2, 0, self._inner(node))

    def _hourglass(self, mpl: SimpleNamespace, ax: Any, x: float, h: float, node: IRNode) -> None:
        waist = h * 0.28
        verts = [
            (x, -h / 2),
            (x, h / 2),
            (x + _SLOT / 2, waist / 2),
            (x + _SLOT, h / 2),
            (x + _SLOT, -h / 2),
            (x + _SLOT / 2, -waist / 2),
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
        self._text(ax, x + _SLOT / 2, 0, self._inner(node))

    def _text(self, ax: Any, x: float, y: float, text: str) -> None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=_FS,
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
            if edge.source_id not in boxes or edge.target_id not in boxes:
                continue
            src, tgt = boxes[edge.source_id], boxes[edge.target_id]
            if edge.kind == EdgeKind.FORWARD:
                self._arrow(mpl, ax, (src[1], 0), (tgt[0], 0), "-|>", "solid", 1.4)
            elif edge.kind == EdgeKind.SKIP:
                skips.append((src, tgt, edge.label))
            elif edge.kind == EdgeKind.RESIDUAL:
                self._arrow(
                    mpl, ax, (src[0], -src[3] / 2), (tgt[1], -tgt[3] / 2), "-|>",
                    "dashed", 1.1, rad=-0.4,
                )
        max_top = _H_MID
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
                    "dashed", 1.1, rad=-0.6,
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
        # arc3 bulges by 0.5 * rad * dist at the apex, so rad = 2 * bulge / dist
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
                fontsize=_FS - 1, color=get_color("pink"), zorder=5,
            )
        return apex + 0.6

    def _draw_zoom(
        self,
        mpl: SimpleNamespace,
        ax: Any,
        boxes: dict[str, tuple[float, float, float, float]],
    ) -> None:
        target = self.graph.node(self.zoom.target)
        layers = [n for n in target.walk() if n.role == NodeRole.LAYER]
        if not layers:
            return
        anchor_id = "/".join(self.zoom.target.split("/")[: self.depth + 1])
        anchor = boxes.get(anchor_id)
        inset = ax.inset_axes(self._inset_bounds())
        inset.set_xlim(0, 1)
        inset.set_ylim(0, len(layers))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor(get_color("gray"))
        for i, layer in enumerate(reversed(layers)):
            y = i + 0.5
            inset.add_patch(
                mpl.FancyBboxPatch(
                    (0.12, y - 0.36),
                    0.76,
                    0.72,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=type_fill(layer.type_label),
                    edgecolor=type_color(layer.type_label),
                    linewidth=1.3,
                )
            )
            inset.text(0.5, y, layer.label, ha="center", va="center", fontsize=_FS - 2,
                       color=get_color("dark"))
            if i:
                inset.annotate(
                    "", xy=(0.5, y - 0.36), xytext=(0.5, y - 0.64),
                    arrowprops=dict(arrowstyle="-|>", color=get_color("darkish")),
                )
        inset.set_title(target.label, fontsize=_FS - 1, color=get_color("dark"))
        if anchor is not None:
            b = anchor
            ax.add_patch(
                mpl.plt.Rectangle(
                    (b[0], -b[3] / 2), _SLOT, b[3], fill=False,
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

    def _inset_bounds(self) -> list[float]:
        size = self.zoom.size
        loc = self.zoom.loc
        x = 1.0 - size if "right" in loc else 0.0
        y = 1.0 - size if "upper" in loc else 0.0
        return [x, y, size, size]

    def _finish(self, mpl: SimpleNamespace, ax: Any, drawables: list[IRNode]) -> None:
        ax.set_aspect("equal")
        ax.axis("off")
        if self.title:
            ax.set_title(self.title, fontsize=_FS + 3, color=get_color("dark"), pad=14)
        if self.show_legend:
            self._legend(mpl, ax, drawables)

    def _legend(self, mpl: SimpleNamespace, ax: Any, drawables: list[IRNode]) -> None:
        seen: dict[str, Any] = {}
        for node in drawables:
            if self.color_labels:
                key = node.label
                fill, stroke = self._fill(node), self._stroke(node)
            elif node.role == NodeRole.COMPONENT:
                key, fill, stroke = (
                    node.label,
                    color_variant(component_color(node), 150),
                    get_color(component_color(node)),
                )
            else:
                key, fill, stroke = (
                    node.type_label,
                    type_fill(node.type_label),
                    type_color(node.type_label),
                )
            if key not in seen:
                seen[key] = mpl.Patch(facecolor=fill, edgecolor=stroke, label=key)
        if seen:
            ax.legend(
                handles=list(seen.values()),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(seen), 4),
                fontsize=_FS - 1,
                frameon=False,
            )

    def _supported_formats(self) -> set[str]:
        """Return writable vector/raster formats."""
        return {"pdf", "svg", "png"}

    def _save_impl(
        self, artifact: Any, filepath: Path, width: int, height: int, scale: int
    ) -> None:
        """Write the figure at its content size; dpi controls raster sharpness."""
        del width, height
        figw = artifact.get_size_inches()[0]
        dpi = min(100 * scale, int(9000 / max(figw, 1.0)))
        artifact.savefig(filepath, dpi=dpi, bbox_inches="tight", transparent=True)
        _require_mpl().plt.close(artifact)

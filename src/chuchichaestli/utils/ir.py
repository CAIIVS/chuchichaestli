# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic semantic intermediate representation (IR) for visualization."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING
import torch
from torch import nn
from chuchichaestli.utils.modules import LayerInfo, info_forward_pass

if TYPE_CHECKING:
    from chuchichaestli.utils.adapters import AdapterRegistry

__all__ = [
    "NodeRole",
    "EdgeKind",
    "GraphLevel",
    "normalize_level",
    "Geometry",
    "IRNode",
    "IREdge",
    "IRGraph",
    "build_ir",
]


class GraphLevel(IntEnum):
    """Abstraction level selected for rendering."""

    COMPONENT = 0
    LEVEL = 1
    BLOCK = 2
    LAYER = 3


def normalize_level(level: int | str | GraphLevel) -> int:
    """Return the node-depth cutoff for a requested abstraction level.

    Args:
        level: Level as int, name, or `GraphLevel`.
    """
    if isinstance(level, str):
        level = GraphLevel[level.upper()]
    return int(level) + 1


class NodeRole(str, Enum):
    """Semantic role of an IR node, ordered by abstraction depth."""

    MODEL = "model"
    COMPONENT = "component"
    LEVEL = "level"
    BLOCK = "block"
    LAYER = "layer"


class EdgeKind(str, Enum):
    """Semantic kind of an IR edge."""

    FORWARD = "forward"
    SKIP = "skip"
    RESIDUAL = "residual"
    MERGE = "merge"


@dataclass
class Geometry:
    """Renderer size hints derived from channel/shape metadata.

    Args:
        channels: Feature-channel count (drives trapezoid thickness).
        spatial: Spatial resolution per dim, e.g. (H, W).
        level_index: 0-based spatial level (0 = highest resolution).
        is_downsample: Whether the node reduces spatial resolution.
        is_upsample: Whether the node increases spatial resolution.
    """

    channels: int | None = None
    spatial: tuple[int, ...] | None = None
    level_index: int | None = None
    is_downsample: bool = False
    is_upsample: bool = False


@dataclass
class IRNode:
    """A semantic node in the visualization hierarchy.

    Args:
        id: Stable structural path, e.g. `model/encoder/level0/block0`.
        role: Semantic role.
        type_label: Human-readable type (from the labelling helpers).
        label: Display label for the renderer.
        module: Backing module, or None for synthetic grouping nodes.
        info: Backing LayerInfo for metadata, or None.
        depth: Abstraction depth; equals `id.count("/")`.
        children: Ordered child nodes.
        geometry: Renderer size hints.
        num_params: Parameter count of this node's subtree.
        meta: Free-form adapter annotations.
    """

    id: str
    role: NodeRole
    type_label: str
    label: str
    module: nn.Module | None = None
    info: LayerInfo | None = None
    depth: int = 0
    children: list[IRNode] = field(default_factory=list)
    geometry: Geometry = field(default_factory=Geometry)
    num_params: int = 0
    meta: dict = field(default_factory=dict)

    def add(self, child: IRNode) -> IRNode:
        """Append a child and return it.

        Args:
            child: Node to append.
        """
        self.children.append(child)
        return child

    def walk(self) -> Iterator[IRNode]:
        """Yield this node then all descendants, pre-order."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class IREdge:
    """A first-class semantic edge between two IR nodes.

    Args:
        source_id: Origin node id.
        target_id: Destination node id.
        kind: Semantic kind.
        label: Optional edge annotation (e.g. "concat").
        meta: Free-form annotations.
    """

    source_id: str
    target_id: str
    kind: EdgeKind
    label: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class IRGraph:
    """Root of the semantic IR: a node tree plus a flat edge list.

    Args:
        root: The MODEL node.
        edges: All edges at all levels.
        index: Map from node id to node (built by `compile`).
    """

    root: IRNode
    edges: list[IREdge] = field(default_factory=list)
    index: dict[str, IRNode] = field(default_factory=dict)

    def compile(self) -> IRGraph:
        """Populate `index` and aggregate subtree parameter counts."""
        self.index = {node.id: node for node in self.root.walk()}
        self._aggregate_params(self.root)
        return self

    @staticmethod
    def _aggregate_params(node: IRNode) -> int:
        # Synthetic group nodes have no info, so they aggregate from children.
        child_sum = sum(IRGraph._aggregate_params(child) for child in node.children)
        node.num_params = node.info.num_params if node.info is not None else child_sum
        return node.num_params

    def node(self, node_id: str) -> IRNode:
        """Look up a node by id.

        Args:
            node_id: Node id to fetch.
        """
        return self.index[node_id]

    def nodes_by_role(self, role: NodeRole) -> list[IRNode]:
        """Return all nodes of a role in document order.

        Args:
            role: Role to filter by.
        """
        return [node for node in self.root.walk() if node.role == role]

    def representative(self, role: NodeRole = NodeRole.BLOCK) -> IRNode | None:
        """Pick a representative node of a role for exemplary zoom.

        Prefers the first node with more than one child (a real multi-layer
        block over a single-layer head like `conv_in`); falls back to the
        first match.

        Args:
            role: Role to find a representative for.
        """
        matches = self.nodes_by_role(role)
        if not matches:
            return None
        return next((n for n in matches if len(n.children) > 1), matches[0])

    def zoom_targets(self) -> list[str]:
        """Node ids suitable as exemplary-zoom targets (nodes with layers)."""
        return [
            node.id
            for node in self.root.walk()
            if node.role != NodeRole.LAYER
            and any(c.role == NodeRole.LAYER for c in node.walk())
        ]

    def _unknown_zoom_error(self, node_id: str) -> ValueError:
        targets = self.zoom_targets()
        hint = "\n  ".join(targets) if targets else "(none)"
        return ValueError(
            f"Unknown zoom target {node_id!r}. Pass zoom=True to auto-pick, or "
            f"one of these node ids:\n  {hint}"
        )

    def view(self, depth: int) -> IRGraph:
        """Return a copy collapsed to a maximum node depth.

        Deeper nodes are pruned; their parameters remain aggregated in the
        surviving boundary node. Edges are rewritten to the nearest surviving
        ancestor and de-duplicated; a residual edge collapsing within a single
        node sets `meta["has_residual"]` there instead.

        Args:
            depth: Maximum node depth to keep (0 = MODEL only).
        """
        return self._project(depth, zoom_id=None)

    def zoom(self, node_id: str, depth: int = 1) -> IRGraph:
        """Return a `view` at `depth` with one node's subtree kept full.

        Args:
            node_id: Node id to expand fully (exemplary zoom).
            depth: Base collapse depth for the rest of the graph.
        """
        if node_id not in self.index:
            raise self._unknown_zoom_error(node_id)
        return self._project(depth, zoom_id=node_id)

    def _survivor(self, node_id: str, depth: int, zoom_id: str | None) -> str:
        if zoom_id is not None and (
            node_id == zoom_id
            or node_id.startswith(zoom_id + "/")
            or zoom_id.startswith(node_id + "/")
        ):
            return node_id
        segments = node_id.split("/")
        if len(segments) - 1 <= depth:
            return node_id
        return "/".join(segments[: depth + 1])

    def _project(self, depth: int, zoom_id: str | None) -> IRGraph:
        def keep(node: IRNode) -> bool:
            return self._survivor(node.id, depth, zoom_id) == node.id

        def clone(node: IRNode) -> IRNode:
            new = IRNode(
                id=node.id,
                role=node.role,
                type_label=node.type_label,
                label=node.label,
                module=node.module,
                info=node.info,
                depth=node.depth,
                geometry=node.geometry,
                num_params=node.num_params,
                meta=dict(node.meta),
            )
            new.children = [clone(c) for c in node.children if keep(c)]
            return new

        root = clone(self.root)
        index = {node.id: node for node in root.walk()}
        edges: list[IREdge] = []
        seen: dict[tuple[str, str, EdgeKind], IREdge] = {}
        for edge in self.edges:
            source = self._survivor(edge.source_id, depth, zoom_id)
            target = self._survivor(edge.target_id, depth, zoom_id)
            if source == target:
                if edge.kind == EdgeKind.RESIDUAL and source in index:
                    index[source].meta["has_residual"] = True
                continue
            if source not in index or target not in index:
                continue
            key = (source, target, edge.kind)
            existing = seen.get(key)
            if existing is None:
                new_edge = IREdge(source, target, edge.kind, edge.label, {"count": 1})
                seen[key] = new_edge
                edges.append(new_edge)
            else:
                existing.meta["count"] += 1
        graph = IRGraph(root=root, edges=edges)
        graph.index = index
        return graph


def build_ir(
    model: nn.Module,
    input_shape: Sequence[int] | torch.Size | None = None,
    input_dtype: torch.dtype = torch.float32,
    registry: AdapterRegistry | None = None,
) -> IRGraph:
    """Build a semantic IR graph for a model.

    Traces the module once (with shapes if `input_shape` is given, structure-only
    otherwise), resolves an adapter, and returns the compiled graph.

    Args:
        model: Model to visualize.
        input_shape: Input shape for shape tracing; structure-only if None.
        input_dtype: Input dtype for tracing.
        registry: Adapter registry (defaults to the built-in one).
    """
    from chuchichaestli.utils.adapters import default_registry

    if input_shape is not None:
        info_list = info_forward_pass(
            model, input_shape=input_shape, input_dtype=input_dtype, use_cache=False
        )
    else:
        info_list = info_forward_pass(model, use_cache=False)
    info_by_id = {info.layer_id: info for info in info_list}
    adapter = (registry or default_registry()).resolve(model)
    return adapter.build(model, info_by_id).compile()

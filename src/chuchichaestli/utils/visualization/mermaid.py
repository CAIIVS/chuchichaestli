# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mermaid backend for converting PyTorch models architectures to mermaid diagrams."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import torch
from torch import nn
from typing import get_args, Literal, Any
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Callable, Sequence
from chuchichaestli.utils import get_layer_type, metric_suffix
from chuchichaestli.utils.visualization import get_color, color_variant
from chuchichaestli.utils.visualization.base import (
    LabelField,
    LABEL_FIELDS,
    TYPE_COLOR,
    AspectSpec,
    normalize_aspect,
)
from chuchichaestli.utils.ir import EdgeKind, IRNode, build_ir

__all__ = ["MermaidDiagram"]


DiagramDirection = Literal[
    "TB",
    "TD",
    "BT",
    "LR",
    "RL",
    "topdown",
    "top-down",
    "top_down",
    "topbottom",
    "top-bottom",
    "top_bottom",
    "downtop",
    "down-top",
    "down_top",
    "bottomtop",
    "bottom-top",
    "bottom_top",
    "leftright",
    "left-right",
    "left_right",
    "rightleft",
    "right-left",
    "right_left",
    "vertical",
    "horizontal",
    "down",
    "up",
    "right",
    "left",
]

_GroupBy = Literal["type", "depth", "level", "block", "encoder_decoder"]

# Base long-edge (inches) for the mermaid page when an aspect is imposed
_MMD_BASE_INCHES = 8.0


def _mermaid_direction(direction: DiagramDirection) -> str | None:
    """Map a recognized direction alias to its mermaid code.

    Args:
        direction: Direction alias (see `DiagramDirection`).

    Returns:
        The mermaid code (`"TB"`/`"BT"`/`"LR"`/`"RL"`), or `None` if the alias
        is not recognized.
    """
    key = direction.lower().strip().replace("-", "").replace("_", "")
    return {
        "tb": "TB",
        "td": "TB",
        "topdown": "TB",
        "topbottom": "TB",
        "vertical": "TB",
        "down": "TB",
        "bt": "BT",
        "downtop": "BT",
        "bottomtop": "BT",
        "up": "BT",
        "lr": "LR",
        "leftright": "LR",
        "horizontal": "LR",
        "right": "LR",
        "rl": "RL",
        "rightleft": "RL",
        "left": "RL",
    }.get(key)


def _mermaid_text(text: str) -> str:
    r"""Normalize text for use inside a quoted mermaid label.

    IR labels may carry a raw newline (e.g. `Reparameterize\n(mu, sigma)`),
    which splits the declaration across lines; `</br>` is mermaid's documented
    line break and keeps one declaration per line.

    Args:
        text: Raw label text.
    """
    return text.replace("\r\n", "</br>").replace("\n", "</br>").replace("\r", "</br>")


def _mermaid_shape_brackets(shape: str) -> tuple[str, str]:
    """Map shapes to the mermaid bracket standard."""
    shape = shape.lower().strip()
    if shape == "stadium":
        return ("([", "])")
    elif shape == "rounded":
        return ("(", ")")
    elif shape == "box":
        return ("[", "]")
    elif shape == "hexagon":
        return ("{{", "}}")
    elif shape == "trapezoid":
        return ("[/", "\\]")
    elif shape == "trapezoid-up":
        return ("[\\", "/]")
    elif shape == "trapezoid-down":
        return ("[/", "\\]")
    elif shape == "cylinder":
        return ("[(", ")]")
    elif shape == "diamond":
        return ("{", "}")
    else:
        return ("([", "])")


@dataclass(frozen=True)
class MermaidClass:
    """A mermaid style class: CSS name, legend text, node shape, and stroke.

    Args:
        name: CSS class name (lowercase); its capitalization is the key used to
            look up a node's style (matches `DEFAULT_MODULE_LABELS` values).
        desc: Human-readable legend description.
        shape: Node shape key (see `_mermaid_shape_brackets`).
        stroke_width: Border width in pixels.
        dashed: Whether the border is dashed (used for concat/merge operations).
        color: Palette colour name; None derives it from the shared type palette
            (`base.TYPE_COLOR`, keyed by the capitalized name).
    """

    name: str
    desc: str
    shape: str
    stroke_width: int = 3
    dashed: bool = False
    color: str | None = None

    @property
    def fill(self) -> str:
        """Palette colour name, derived from `TYPE_COLOR` when not set."""
        return self.color or TYPE_COLOR.get(self.name.capitalize(), "grey")

    @property
    def props(self) -> str:
        """The `classDef` stroke/colour suffix."""
        dash = ",stroke-dasharray: 5 5" if self.dashed else ""
        return (
            f"stroke-width:{self.stroke_width}px{dash}"
            f",color:{color_variant('dark', shift=-30)}"
        )

    @property
    def style(self) -> str:
        """The full mermaid `classDef` style string."""
        return f"fill:{color_variant(self.fill, 150)},stroke:{get_color(self.fill)},{self.props}"


MermaidClasses = [
    MermaidClass("conv", "Convolutional Layer", "stadium"),
    MermaidClass("linear", "Linear/Dense Layer", "stadium"),
    MermaidClass("norm", "Normalization Layer", "stadium", stroke_width=2),
    MermaidClass("dropout", "Dropout Layer", "rounded", stroke_width=2),
    MermaidClass("upsample", "Upsampling Layer", "trapezoid-up"),
    MermaidClass("downsample", "Downsampling Layer", "trapezoid-down"),
    MermaidClass("activation", "Activation Function", "rounded", stroke_width=2),
    MermaidClass("attention", "Attention Mechanism", "stadium"),
    MermaidClass("embedding", "Embedding Layer", "stadium"),
    MermaidClass("recurrent", "Recurrent Layer (LSTM/GRU)", "stadium"),
    MermaidClass("concat", "Concatentation", "diamond", stroke_width=4, dashed=True, color="golden"),
    MermaidClass("merge", "Merge/Add Operation", "diamond", stroke_width=4, dashed=True, color="brown"),
    MermaidClass("default", "Layer", "stadium", color="grey"),
]

MermaidStyleClasses = {c.name: c.style for c in MermaidClasses}

MermaidLayerStyles = {
    c.name.capitalize(): {"shape": c.shape, "class": c.name, "desc": c.desc}
    for c in MermaidClasses
}


class MermaidDiagram:
    """Convert PyTorch modules to mermaid diagrams with flexible abstraction.

    Note:
        The diagram generation might work on other PyTorch modules, but is not guaranteed.
    """

    def __init__(
        self,
        model: nn.Module,
        trace_forward: bool = True,
        input_shape: Sequence[int] | torch.Size | None = None,
        input_dtype: torch.dtype = torch.float32,
        auto: bool = True,
        direction: DiagramDirection = "horizontal",
        group_direction: DiagramDirection = "vertical",
        max_depth: int | None = None,
        group_by: _GroupBy | list[_GroupBy] | None = None,
        type_map: dict[str, str] | None = None,
        class_fn: Callable[[nn.Module], str | None] | None = None,
        layer_styles: dict[str, dict[str, str]] | None = None,
        style_classes: dict[str, str] | None = None,
        label_fields: Sequence[LabelField] = ("name", "params"),
        show_legend: bool = False,
    ):
        """Constructor.

        Args:
            model: Model instance from chuchichaestli (or alternative PyTorch model).
            trace_forward: Whether to trace forward pass to detect functional connections.
            input_shape: Input tensor shape for forward tracing, e.g. `(1, 3, 256, 256)`.
            input_dtype: Input tensor type for forward tracing (default: `torch.float32`).
            auto: If `True`, graph structure will be extracted upon initialization.
            direction: Diagram direction, e.g. T(op)D(own), L(eft)R(ight), etc.
            group_direction: Direction within grouped subgraphs.
            max_depth: Maximum depth for module recursion; for `None` full depth is used.
            group_by: Strategy for grouping layers into subgraphs.
              Options:
                  - 'type': Group by layer type
                  - 'depth': Group by graph depth.
                  - 'level': Group by resolution level
                  - 'block': Group layers under their parent block (at
                    max_depth=4, blocks become subgraphs of their layers).
                  - 'encoder_decoder': Group by architectural functionality.
                  - None: No grouping (default)
                  - list: Nest several strategies outer->inner, e.g.
                    `["encoder_decoder", "block"]` nests block subgraphs
                    inside encoder/decoder subgraphs.
            type_map: Mapping from default layer types to custom names.
            class_fn: Function to categorize unknown layers (or alternative to default layers).
            layer_styles: Additional layer style definitions.
            style_classes: Additional CSS style class definitions.
            label_fields: Which fields to label nodes with (shared with the
                matplotlib backend), any of `"name"`, `"channels"`,
                `"resolution"`, `"params"` (`"channels"`/`"resolution"`
                both show the tensor shape; `"kernel"` is not rendered here).
            show_legend: Whether to show the legend explaining components.
        """
        unknown = [f for f in label_fields if f not in LABEL_FIELDS]
        if unknown:
            raise ValueError(
                f"Unknown label field(s) {unknown}; choose from {list(LABEL_FIELDS)}"
            )
        self.model = model
        self.trace_forward = trace_forward
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.direction = self.parse_direction(direction)
        self.group_direction = self.parse_direction(group_direction)
        self.max_depth = max_depth
        self.label_fields = tuple(label_fields)
        self.show_names = "name" in self.label_fields
        self.show_params = "params" in self.label_fields
        self.show_shapes = bool(set(self.label_fields) & {"channels", "resolution"})
        self.show_legend = show_legend
        self.group_by = group_by
        self.type_map = type_map or {}
        self.class_fn = class_fn or (lambda args: None)

        # mermaid styles and classes
        self.layer_styles: dict[str, dict[str, str]] = MermaidLayerStyles.copy()
        if layer_styles:
            self.layer_styles.update(layer_styles)
        self.style_classes: dict[str, str] = MermaidStyleClasses.copy()
        if style_classes:
            self.style_classes.update(style_classes)

        # model content analysis
        self.model_graph: Any = None
        self._ir = None
        self._nodes: list[dict[str, Any]] = []
        self._edges: list[tuple[str, str, str | None]] = []
        self._subgraphs = defaultdict(list)
        self._group_labels: dict[str, str] = {}
        self._group_paths: dict[str, list[str]] = {}
        if auto:
            self.extract_model_graph()
            self._aggregate_components()

    @staticmethod
    def parse_direction(direction: DiagramDirection) -> str:
        """Parse direction string and convert to mermaid format.

        Args:
            direction: Diagram direction, e.g. T(op)D(own), L(eft)R(ight), etc.

        Returns:
            Mermaid direction ['TB', 'BT', 'LR', 'RL']

        Raises:
            ValueError: If direction is not recognized.
        """
        mmd_dir = _mermaid_direction(direction)
        if mmd_dir is not None:
            return mmd_dir
        raise ValueError(
            f"Unknown direction: '{direction}'. "
            f"Valid options are {MermaidDiagram.list_directions()}."
        )

    @staticmethod
    def list_directions() -> list[str]:
        """List all valid direction string inputs that can be converted into mermaid directions.

        Returns:
            List of valid direction strings.
        """
        return list(get_args(DiagramDirection))

    @staticmethod
    def list_default_style_classes() -> list[tuple[str, str]]:
        """List all default class styles for mermaid diagrams."""
        return list(MermaidStyleClasses.copy().items())

    @staticmethod
    def list_default_layer_styles() -> list[tuple[str, dict[str, str]]]:
        """List all layer styles for mermaid diagrams."""
        return list(MermaidLayerStyles.copy().items())

    @staticmethod
    def cli_available() -> bool:
        """Check if mermaid CLI (mmdc) is installed and available.

        Returns:
            True if mmdc is available on the host, otherwise False.
        """
        return shutil.which("mmdc") is not None

    @staticmethod
    def check_cli_version() -> str | None:
        """Get the version of the installed mermaid CLI (if available).

        Returns:
            Version string of the mermaid CLI or None
        """
        if not MermaidDiagram.cli_available():
            return None

        try:
            result = subprocess.run(
                ["mmdc", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception as e:
            raise RuntimeError(e)

    def extract_model_graph(
        self, input_shape: Sequence[int] | torch.Size | None = None
    ):
        """Build the semantic IR graph from the model."""
        if input_shape is None:
            input_shape = self.input_shape
        self._ir = build_ir(
            self.model,
            input_shape=input_shape if self.trace_forward else None,
            input_dtype=self.input_dtype,
        )
        self.model_graph = self._ir

    def _aggregate_components(self):
        """Build nodes, edges, and subgraphs from the semantic IR."""
        if self._ir is None:
            return
        self._group_labels = {}
        self._group_paths = {}
        # `None` renders the full model: the deepest node depth keeps every node.
        cutoff = (
            self.max_depth
            if self.max_depth is not None
            else max(n.depth for n in self._ir.root.walk())
        )
        view = self._ir.view(cutoff)
        drawables = [
            n for n in view.root.walk() if n.depth == cutoff or not n.children
        ]
        idmap: dict[str, str] = {}
        for node in drawables:
            node_id = self._sanitize_mermaid_id(node.id)
            idmap[node.id] = node_id
            self._nodes.append(
                {
                    "id": node_id,
                    "type": self._ir_type(node),
                    "label": self._ir_label(node),
                    "ir": node,
                }
            )
            keys = self._ir_group_keys(node)
            self._group_paths[node_id] = keys
            for key in keys:
                self._group_labels.setdefault(key, self._ir_group_label(key))
            if keys:
                self._subgraphs[keys[-1]].append(node_id)
        for edge in view.edges:
            # Group endpoints are not drawn at layer depth;
            # resolve them onto the drawn frontier.
            source_id = view.on_frontier(edge.source_id, idmap, last=True)
            target_id = view.on_frontier(edge.target_id, idmap, last=False)
            if source_id is None or target_id is None or source_id == target_id:
                continue
            src, tgt = idmap[source_id], idmap[target_id]
            if edge.kind == EdgeKind.SKIP:
                self._edges.append((src, tgt, "skip"))
            elif edge.kind == EdgeKind.RESIDUAL:
                self._edges.append((src, tgt, "residual"))
            else:
                self._edges.append((src, tgt, None))

    def _reload_components(self) -> None:
        """Discard the derived nodes/edges/subgraphs and rebuild them."""
        self._nodes = []
        self._edges = []
        self._subgraphs = defaultdict(list)
        self._aggregate_components()

    def nodes(self, _reload: bool = False) -> list[dict[str, Any]]:
        """Get nodes from the model graph.

        Args:
            _reload: If `True`, reload the nodes from the model graph.
        """
        if _reload or not self._nodes:
            self._reload_components()
        return self._nodes

    def edges(self, _reload: bool = False) -> list[tuple[str, str, str | None]]:
        """Get edges from the model graph.

        Args:
            _reload: If `True`, reload the nodes from the model graph.
        """
        if _reload or not self._edges:
            self._reload_components()
        return self._edges

    def subgraphs(self, _reload: bool = False) -> dict[str, list[str]]:
        """Get subgraph groupings.

        Args:
            _reload: If `True`, reload the nodes from the model graph.
        """
        if _reload or not self._subgraphs:
            self._reload_components()
        return dict(self._subgraphs)

    def _ir_type(self, node: IRNode) -> str:
        """Mermaid style-class key for an IR node."""
        label = self.class_fn(node.module) if node.module is not None else None
        label = label or node.type_label
        if label not in self.layer_styles and node.module is not None:
            label = get_layer_type(node.module)
        return self.type_map.get(label, label)

    def set_type_name(self, default_type: str, type_renamed: str):
        """Change default labelling.

        Args:
            default_type: Type label to rename.
            type_renamed: Name to use in its place.
        """
        self.type_map[default_type] = type_renamed
        # Node types are materialized on construction; re-derive them so the
        # new name reaches an already-built diagram.
        self._reload_components()

    def _sanitize_mermaid_id(
        self,
        name: str,
        layer_id: int | None = None,
    ) -> str:
        """Convert a module name to a valid mermaid ID."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and not sanitized[0].isalpha():
            sanitized = "node_" + sanitized
        if layer_id is not None:
            sanitized = f"{sanitized}_{layer_id}"
        return sanitized or "node"

    def _ir_label(self, node: IRNode) -> str:
        """Node label with optional parameter count and output shape."""
        labels = []
        if self.show_names:
            labels.append(node.label)
        if self.show_params and node.num_params > 0:
            labels.append(f"{metric_suffix(node.num_params, 1)} params")
        if self.show_shapes and node.info is not None and node.info.output_size:
            labels.append(str(node.info.output_size))
        return "</br>".join(labels)

    def _ir_group_keys(self, node: IRNode) -> list[str]:
        """Ordered subgraph keys (outer->inner) for the active strategies."""
        strategies = self.group_by if isinstance(self.group_by, list) else [self.group_by]
        keys = []
        for strategy in strategies:
            key = self._ir_group_key(node, strategy)
            if key:
                keys.append(key)
        return keys

    def _ir_group_key(self, node: IRNode, strategy: str | None) -> str | None:
        """Subgraph key for an IR node under a single grouping strategy."""
        if strategy == "type":
            return self._ir_type(node)
        if strategy == "depth":
            return f"Depth {node.depth}"
        if strategy == "level":
            idx = node.geometry.level_index
            return f"Level {idx}" if idx is not None else None
        if strategy == "block":
            parts = node.id.split("/")
            return "/".join(parts[:4]) if len(parts) >= 4 else node.id
        if strategy == "encoder_decoder":
            side = node.id.split("/")[1] if "/" in node.id else ""
            return {
                "encoder": "Encoder",
                "decoder": "Decoder",
                "bottleneck": "Bottleneck",
                "latent": "Latent",
            }.get(side, "I/O")
        return None

    def _ir_group_label(self, key: str) -> str:
        """Display title for a subgraph key (a node id maps to its label)."""
        node = self._ir.index.get(key) if self._ir is not None else None
        return node.label if node is not None else key

    def _subgraph_tree(self) -> dict[str, Any]:
        """Nest nodes into a tree following each node's group-key path."""
        root: dict[str, Any] = {"children": {}, "nodes": []}
        for node in self._nodes:
            cursor = root
            for key in self._group_paths.get(node["id"], []):
                cursor = cursor["children"].setdefault(
                    key, {"children": {}, "nodes": []}
                )
            cursor["nodes"].append(node["id"])
        return root

    def _emit_group_nodes(self, node_ids: list[str], indent: str) -> list[str]:
        """Emit node declarations + style classes at the given indent."""
        lines = []
        for node_id in node_ids:
            node = next(n for n in self._nodes if n["id"] == node_id)
            shape = self._get_node_shape(node["type"], node["label"])
            lines.append(f"{indent}{node['id']}{shape}")
            layer_style = self.layer_styles.get(
                node["type"], self.layer_styles.get("Default", {})
            )
            css_class = layer_style.get("class", "default")
            lines.append(f"{indent}class {node['id']} {css_class}")
        return lines

    def _emit_group_tree(
        self, tree: dict[str, Any], indent: str, prefix: str = ""
    ) -> list[str]:
        """Recursively emit nested subgraphs, then this level's direct nodes."""
        lines = []
        for key, sub in tree["children"].items():
            gid = self._sanitize_mermaid_id(f"{prefix}{key}")
            title = self._group_labels.get(key, key)
            lines.append(f'{indent}subgraph {gid}["{_mermaid_text(title)}"]')
            lines.append(f"{indent}    direction {self.group_direction}")
            lines.extend(self._emit_group_tree(sub, indent + "    ", f"{prefix}{key}/"))
            lines.append(f"{indent}end")
            lines.append("")
        lines.extend(self._emit_group_nodes(tree["nodes"], indent))
        return lines

    def _get_node_shape(self, layer_type: str, name: str) -> str:
        """Get the mermaid shape syntax for a node."""
        style = self.layer_styles.get(layer_type, self.layer_styles.get("Default", {}))
        shape = style.get("shape", "stadium")
        brackets = _mermaid_shape_brackets(shape)
        return f'{brackets[0]}"{_mermaid_text(name)}"{brackets[1]}'

    def generate_configs(
        self,
        theme: str = "dark",
        variables: dict[str, str] = {
            "primaryTextColor": f"{color_variant('dark', shift=-30)}",
        },
    ) -> list[str]:
        """Generate configs defining global diagram settings."""
        lines = []
        lines.append("---")
        lines.append("config:")
        lines.append(f"  theme: '{theme}'")
        lines.append("  themeVariables:")
        for k, v in variables.items():
            lines.append(f"    {k}: '{v}'")
        lines.append("---")
        return lines

    def generate_legend(self) -> list[str]:
        """Generate legend explaining the diagram components."""
        lines = []
        lines.append("    %% Legend")
        lines.append("    subgraph Legend")
        lines.append("        direction LR")

        # Get unique layer types from the model
        unique_types = set(node["type"] for node in self.nodes())

        # Add example nodes for types that appear in the model
        for layer_type in sorted(unique_types):
            if layer_type in self.layer_styles:
                node_id = f"legend_{layer_type.lower()}"
                style_info = self.layer_styles[layer_type]
                shape = self._get_node_shape(layer_type, style_info["desc"])
                lines.append(f"        {node_id}{shape}")

                # Get the CSS class
                css_class = style_info.get("class", layer_type.lower())
                lines.append(f"        class {node_id} {css_class}")

        lines.append("    end")
        lines.append("")

        # Add legend notes
        lines.append("    %% Legend Notes:")
        lines.append("    %% → : Standard connection")
        lines.append("    %% -.-> : Skip connection")
        lines.append("    %% -->|name| : Labeled connection")
        lines.append("    %% ⬥ Diamond shapes indicate Concat/Merge operations")
        lines.append(
            "    %% ⬭ Stadium shapes represent layer slabs (good for U-Net visualization)"
        )
        lines.append("")
        return lines

    def generate(self) -> str:
        """Generate a mermaid diagram string."""
        if not self.model_graph:
            self.extract_model_graph()

        if not self._nodes:
            self._aggregate_components()

        # Default config
        lines = self.generate_configs()

        # Start graph structure
        lines.append(f"graph {self.direction}")
        # Add style class definintions
        lines.append("    %% Style definitions")
        for css_class, style_def in self.style_classes.items():
            lines.append(f"    classDef {css_class} {style_def}")
        lines.append("")

        # Add nodes, nested in subgraphs per the (possibly multi-level) grouping
        tree = self._subgraph_tree()
        if tree["children"]:
            lines.append("    %% Grouped layers")
            lines.extend(self._emit_group_tree(tree, "    "))
        else:
            lines.append("    %% Model architecture")
            lines.extend(self._emit_group_nodes(tree["nodes"], "    "))
            lines.append("")

        # Add edges
        lines.append("    %% Connections")
        for edge in self._edges:
            if len(edge) == 3 and edge[2]:
                from_node, to_node, label = edge
                if label in ["skip", "residual"]:
                    lines.append(f"    {from_node} -.->|{label}| {to_node}")
                else:
                    lines.append(f"    {from_node} -->|{label}| {to_node}")
            else:
                lines.append(f"    {edge[0]} --> {edge[1]}")
        lines.append("")

        if self.show_legend:
            lines.extend(self.generate_legend())

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation returns the generated mermaid diagram."""
        return self.generate()

    def save(
        self,
        filename: Path | str,
        dpi: int = 300,
        aspect: AspectSpec | None = None,
    ):
        """Save the diagram to file.

        Args:
            filename: Output path; the suffix selects the format (mmd/svg/png/pdf).
            dpi: Target output resolution (image formats); mapped to the mermaid
                CLI density and page pixels.
            aspect: Target width/height ratio as a `w/h` float or `(w, h)` pair
                sizing the page (image formats); None keeps the natural layout.
        """
        diagram = self.generate()
        filepath = Path(filename)
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        _format = filepath.suffix[1:]
        if _format == "mmd":
            with filepath.open("w", encoding="utf-8") as f:
                f.write(diagram)
        elif _format in ("svg", "png", "pdf"):
            if not self.cli_available():
                raise RuntimeError(
                    "Mermaid CLI (mmdc) is required for image export. "
                    "Visit https://github.com/mermaid-js/mermaid-cli for details."
                )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mmd", delete=False, encoding="utf-8"
            ) as tmpf:
                tmpf.write(diagram)
                tmpf_path = tmpf.name
            try:
                subprocess.run(
                    [
                        "mmdc",
                        "-i",
                        tmpf_path,
                        "-o",
                        str(filepath),
                        *self._mmdc_size_args(dpi, normalize_aspect(aspect)),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Mermaid CLI conversion failed {e.stderr}") from e
            finally:
                try:
                    os.remove(tmpf_path)
                except OSError:
                    pass
        else:
            raise ValueError(
                f"Unsupported file format: {_format}.Use 'mmd', 'svg', 'png', or 'pdf'."
            )
        return filepath

    @staticmethod
    def _mmdc_size_args(dpi: int, aspect: float | None) -> list[str]:
        """Mermaid-CLI size flags from a dpi and optional `w/h` aspect.

        The CLI page is expressed in CSS pixels (96 per inch) and the density
        (`-s`) carries the dpi, so the rendered raster is `inches * dpi` per
        side. Without an aspect the natural layout is kept and only scaled.

        Args:
            dpi: Target output resolution.
            aspect: Target `w/h` ratio (already normalized), or None.
        """
        scale = max(1.0, dpi / 96)
        args = ["-s", f"{scale:g}"]
        if aspect is not None:
            long_in = _MMD_BASE_INCHES
            w_in, h_in = (
                (long_in, long_in / aspect)
                if aspect >= 1
                else (long_in * aspect, long_in)
            )
            args += ["-w", str(round(w_in * 96)), "-H", str(round(h_in * 96))]
        return args


if __name__ == "__main__":
    from pprint import pprint
    from chuchichaestli.models.unet import UNet

    model = UNet(
        dimensions=2,
        in_channels=3,
        n_channels=64,
        out_channels=3,
        down_block_types=("DownBlock",) * 4,
        up_block_types=("UpBlock", "UpBlock", "AttnUpBlock", "AttnUpBlock"),
        block_out_channel_mults=(1, 2, 2, 4),
        res_act_fn="prelu",
        res_dropout=0.4,
        attn_n_heads=2,
        skip_connection_action="concat",
    )

    mmd = MermaidDiagram(
        model,
        direction="LR",
        group_by=None,
        label_fields=("name",),
        show_legend=True,
        max_depth=2,
    )
    if mmd.cli_available():
        print("mmdc version:")
        print(mmd.check_cli_version())
        print()
        print("Mermaid classes:")
        pprint(mmd.list_default_style_classes())
        print()
        print("Mermaid styles:")
        pprint(mmd.list_default_layer_styles())
        print()
        print("Model graph:")
        for info in mmd.model_graph[:4]:
            print(info())
        print()
        print("Mermaid components:")
        pprint(mmd.nodes())
        print()
        print("Generate legend:")
        pprint(mmd.generate_legend())
        print()
        print("Generate diagram:")
        print(mmd.generate())
        print()
        print("Save diagram:")
        print(mmd.save("mermaid_diagram.png", dpi=300, aspect=(3, 2)))
        # print("Mermaid diagram string:")
        # print(mmd)

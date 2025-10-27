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
from collections import defaultdict
from collections.abc import Callable, Sequence
from chuchichaestli.utils import (
    info_forward_pass,
    get_chuchichaestli_block_type,
    get_layer_type,
    metric_suffix,
)
from chuchichaestli.utils.modules import LayerInfo
from chuchichaestli.utils.visualization import get_color, color_variant

__all__ = ["MermaidDiagram", "mermaid_diagram"]


DiagramDirections = Literal[
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


def _mermaid_direction(direction: DiagramDirections) -> str:
    """Map directions to the mermaid standard."""
    direction = direction.lower().strip()
    if direction == "left":
        return "RL"
    elif direction in ("right", "horizontal") or direction.startswith("l"):
        return "LR"
    elif direction in ("down", "vertical") or direction.startswith("t"):
        return "TB"
    elif direction == "up" or direction.startswith("b"):
        return "BT"
    elif direction.startswith("r"):
        return "RL"
    else:
        return "LR"


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


MermaidClasses = [
    "conv",
    "linear",
    "norm",
    "dropout",
    "pool",
    "upsample",
    "downsample",
    "activation",
    "attention",
    "embedding",
    "recurrent",
    "concat",
    "merge",
    "default",
]

MermaidClassDescriptions = [
    "Convolutional Layer",
    "Linear/Dense Layer",
    "Normalization Layer",
    "Dropout Layer",
    "Pooling Layer",
    "Upsampling Layer",
    "Downsampling Layer",
    "Activation Function",
    "Attention Mechanism",
    "Embedding Layer",
    "Recurrent Layer (LSTM/GRU)",
    "Concatentation",
    "Merge/Add Operation",
    "Layer",
]

MermaidClassColors = [
    "green",
    "blue",
    "orange",
    "red",
    "purple",
    "turquoise",
    "marguerite",
    "pink",
    "cyan",
    "yellow",
    "purpleblue",
    "golden",
    "brown",
    "dark",
]

MermaidClassShapes = [
    "stadium",
    "stadium",
    "stadium",
    "rounded",
    "stadium",
    "trapezoid-up",
    "trapezoid-down",
    "rounded",
    "stadium",
    "stadium",
    "stadium",
    "diamond",
    "diamond",
    "stadium",
]

MermaidClassProps = [
    "stroke-width:3px",
    "stroke-width:3px",
    "stroke-width:2px",
    "stroke-width:2px",
    "stroke-width:3px",
    "stroke-width:3px",
    "stroke-width:3px",
    "stroke-width:2px",
    "stroke-width:3px",
    "stroke-width:3px",
    "stroke-width:3px",
    "stroke-width:4px,stroke-dasharray: 5 5",
    "stroke-width:4px,stroke-dasharray: 5 5",
    "stroke-width:3px",
]

MermaidStyleClasses = {
    cls: f"fill:{color_variant(clr, 150)},stroke:{get_color(clr)},{props}"
    for cls, clr, props in zip(
        MermaidClasses,
        MermaidClassColors,
        MermaidClassProps,
    )
}

MermaidLayerStyles = {
    cls.capitalize(): {"shape": s, "class": cls, "desc": d}
    for cls, s, d in zip(
        MermaidClasses,
        MermaidClassShapes,
        MermaidClassDescriptions,
    )
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
        direction: DiagramDirections = "horizontal",
        group_direction: DiagramDirections = "vertical",
        max_depth: int | None = None,
        positions: dict | None = None,
        group_by: dict | None = None,
        type_map: dict[str, str] | None = None,
        class_fn: Callable[[nn.Module], str | None] | None = None,
        layer_styles: dict[str, dict[str, str]] | None = None,
        style_classes: dict[str, str] | None = None,
        show_names: bool = True,
        show_params: bool = True,
        show_shapes: bool = False,
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
            max_depth: Maximum depth for module recursion; for `None` full depth is used.
            positions: Mapping of node IDs to custom positions.
            group_by: Strategy for grouping layers into subgraphs.
              Options:
                  - 'layer': Group by first-level module names
                  - 'type': Group by layer type
                  - None: No grouping (default)
            type_map: Mapping from default layer types to custom names.
            class_fn: Function to categorize unknown layers (or alternative to default layers).
            layer_styles: Additional layer style definitions.
            style_classes: Additional CSS style class definitions.
            show_names: Whether to show the layer names.
            show_params: Whether to show parameter counts of layers.
            show_shapes: Whether to show tensor shapes.
            show_legend: Whether to shwo the legend explaining components.
        """
        self.model = model
        self.trace_forward = trace_forward
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.direction = self.parse_direction(direction)
        self.group_direction = self.parse_direction(group_direction)
        self.max_depth = max_depth
        self.positions = positions or {}
        self.show_names = show_names
        self.show_params = show_params
        self.show_shapes = show_shapes
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
        self.model_graph: list[LayerInfo] = []
        self._nodes: list[dict[str, Any]] = []
        self._edges: list[tuple[str, str, str | None]] = []
        self._subgraphs = defaultdict(list)
        if auto:
            self.extract_model_graph()
            self._aggregate_components()

    @staticmethod
    def parse_direction(direction: DiagramDirections) -> str:
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
        return list(get_args(DiagramDirections))

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
        """Construct info graph from model."""
        if input_shape is None:
            input_shape = self.input_shape
        if self.trace_forward:
            if input_shape is None:
                graph = info_forward_pass(
                    self.model, size=32, input_dtype=self.input_dtype
                )
            else:
                graph = info_forward_pass(
                    self.model, input_shape=input_shape, input_dtype=self.input_dtype
                )
        else:
            graph = info_forward_pass(self.model)
        self.model_graph = graph

    def _aggregate_components(self):
        """Build nodes and edges from the model graph."""
        if not self.model_graph:
            return

        filtered_layers = [
            layer for layer in self.model_graph if self._layer_filter(layer)
        ]

        for i, layer_info in enumerate(filtered_layers):
            layer_type = self._get_layer_type(layer_info.module)
            node_id = self._sanitize_mermaid_id(layer_info.name, layer_info.layer_id)
            label = self._create_node_name(layer_info)
            node = {
                "id": node_id,
                "type": layer_type,
                "label": label,
                "layer_info": layer_info,
            }
            self._nodes.append(node)

            group_key = self._get_group_key(layer_info)
            if group_key:
                self._subgraphs[group_key].append(node_id)

            if i > 0:
                prev_node = self._nodes[i - 1]
                if layer_info.depth < prev_node["layer_info"].depth:
                    self._edges.append((prev_node["id"], node_id, "skip"))
                else:
                    self._edges.append((prev_node["id"], node_id, None))

    def nodes(self, _reload: bool = False) -> list[dict[str, Any]]:
        """Get nodes from the model graph.

        Args:
            _reload: If `True`, reload the nodes from the model graph.
        """
        if _reload or not self._nodes:
            self._nodes = []
            self._edges = []
            self._subgraphs = defaultdict(list)
            self._aggregate_components()
        return self._nodes

    def edges(self, _reload: bool = False) -> list[tuple[str, str, str | None]]:
        """Get edges from the model graph.

        Args:
            _reload: If `True`, reload the nodes from the model graph.
        """
        if _reload or not self._edges:
            self._nodes = []
            self._edges = []
            self._subgraphs = defaultdict(list)
            self._aggregate_components()
        return self._edges

    def subgraphs(self, _reload: bool = False) -> dict[str, list[str]]:
        """Get subgraph groupings.

        Args:
            _reload: If `True`, reload the nodes from the model graph.
        """
        if _reload or not self._subgraphs:
            self._nodes = []
            self._edges = []
            self._subgraphs = defaultdict(list)
            self._aggregate_components()
        return dict(self._subgraphs)

    def _layer_filter(self, layer_info: LayerInfo) -> bool:
        """Determine if a layer should be included in diagram based on depth criteria."""
        # only show leaves if no max depth set
        if self.max_depth is None:
            return layer_info.is_leaf_layer

        # exclude everything beyond max depth
        if self.max_depth < layer_info.depth:
            return False

        # include layers exactly at max depth (target visualization level)
        if layer_info.depth == self.max_depth:
            return True

        # for layer above max depth, only include if they have no children beyond max depth
        if layer_info.children:
            has_visible_children = any(
                child.depth <= self.max_depth for child in layer_info.children
            )
            return not has_visible_children

        return True

    def _get_layer_type(self, module: nn.Module) -> str:
        """Determine the layer type from a PyTorch module class."""
        if self.class_fn:
            layer_type = self.class_fn(module)
        if layer_type is None:
            layer_type = get_chuchichaestli_block_type(module)
        if layer_type is None:
            layer_type = get_layer_type(module)
        if layer_type in self.type_map:
            return self.type_map[layer_type]
        return layer_type

    def set_type_name(self, default_type: str, type_renamed: str):
        """Change default labelling."""
        self.type_map[default_type] = type_renamed

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

    def _create_node_name(self, layer_info: LayerInfo) -> str:
        """Create a name for a node with optional parameter count."""
        labels = []
        if self.show_names:
            labels.append(layer_info.get_layer_name(show_name=False, show_depth=False))
        if self.show_params and layer_info.num_params > 0:
            labels.append(f"{self._format_number(layer_info.num_params)} params")
        if self.show_shapes and layer_info.input_size:
            labels.append(f"{str(layer_info.input_size)}")
        return "</br>".join(labels)

    def _format_number(self, num: int) -> str:
        """Format large numbers with metric suffixes."""
        return metric_suffix(num, 1)

    def _get_group_key(self, layer_info: LayerInfo) -> str:
        """Determine the group key for a module based on grouping strategy."""
        if self.group_by == "type":
            return self._get_layer_type(layer_info.module)
        elif self.group_by == "depth":
            return f"depth_{layer_info.depth}"
        elif self.group_by == "encoder_decoder":
            name_lower = layer_info.name.lower()
        else:
            return None

    def _get_node_shape(self, layer_type: str, name: str) -> str:
        """Get the mermaid shape syntax for a node."""
        style = self.layer_styles.get(layer_type, self.layer_styles.get("Default", {}))
        shape = style.get("shape", "stadium")
        brackets = _mermaid_shape_brackets(shape)
        return f'{brackets[0]}"{name}"{brackets[1]}'

    def generate_configs(
        self,
        theme: str = "dark",
        variables: dict[str, str] = {
            "primaryTextColor": f"{color_variant('dark', shift=-30)}"
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

    def generate(self, auto_connect: bool = True) -> str:
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

        # Add groups instead of nodes
        if self._subgraphs:
            lines.append("    %% Grouped layers")
            for group_name, node_ids in self._subgraphs.items():
                sanitized_group = self._sanitize_mermaid_id(group_name)
                lines.append(f'    subgraph {sanitized_group}["{group_name}"]')
                lines.append(f"       direction {self.group_direction}")
                for node_id in node_ids:
                    # insert node
                    node = next(n for n in self._nodes if n["id"] == node_id)
                    shape = self._get_node_shape(node["type"], node["label"])
                    lines.append(f"        {node['id']}{shape}")
                    # style node
                    layer_style = self.layer_styles.get(
                        node["type"], self.layer_styles.get("Default", {})
                    )
                    css_class = layer_style.get("class", "default")
                    lines.append(f"        class {node['id']} {css_class}")
                lines.append("    end")
                lines.append("")
        # Add all nodes
        else:
            lines.append("    %% Model architecture")
            for node in self._nodes:
                # insert node
                shape = self._get_node_shape(node["type"], node["label"])
                lines.append(f"    {node['id']}{shape}")
                # style node
                layer_style = self.layer_styles.get(
                    node["type"], self.layer_styles.get("Default", {})
                )
                css_class = layer_style.get("class", "default")
                lines.append(f"        class {node['id']} {css_class}")
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
        width: int = 1920,
        height: int = 1080,
        scale: int = 4,
    ):
        """Save the diagram to file.

        Args:
            filepath: Path to save the diagram
        """
        diagram = self.generate()
        filepath = Path(filename)
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        _format = filepath.suffix[1:]
        if _format == "mmd":
            with filepath.open("w") as f:
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
                result = subprocess.run(
                    [
                        "mmdc",
                        "-i",
                        tmpf_path,
                        "-o",
                        filepath,
                        "-w",
                        str(width),
                        "-H",
                        str(height),
                        "-s",
                        str(scale),
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


def mermaid_diagram(
    model: nn.Module,
    direction: DiagramDirections = "horizontal",
    max_depth: int | None = None,
    positions: dict | None = None,
    show_params: bool = True,
    show_shapes: bool = False,
    show_legend: bool = False,
    **kwargs,
) -> MermaidDiagram:
    """Create a mermaid diagram from a model.

    Args:
        model: Model instance from chuchichaestli.
        direction: Diagram direction, e.g. T(op)D(own), L(eft)R(ight), etc.
        max_depth: Maximum depth for module recursion; for `None` full depth is used.
        positions: Mapping of node IDs to custom positions.
        show_params: Whether to show parameter counts of layers.
        show_shapes: Whether to show tensor shapes.
        show_legend: Whether to shwo the legend explaining components.
        kwargs: Other keyword arguments for `MermaidDiagram`.

    Note:
        The diagram generation might work on other PyTorch modules, but is not guaranteed.
    """
    return MermaidDiagram(
        model,
        direction=direction,
        positions=positions,
        max_depth=max_depth,
        show_params=show_params,
        show_shapes=show_shapes,
        show_legend=show_legend,
        **kwargs,
    )


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

    mmd = mermaid_diagram(
        model,
        direction="LR",
        show_names=True,
        show_params=False,
        show_shapes=False,
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
        print(mmd.generate(auto_connect=True))
        print()
        print("Save diagram:")
        print(mmd.save("mermaid_diagram.png", 1080, 720, 4))
        # print("Mermaid diagram string:")
        # print(mmd)

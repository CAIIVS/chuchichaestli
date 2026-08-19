# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Renderer contract and shared styling for model-visualization backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any
from chuchichaestli.utils.modules import DEFAULT_MODULE_LABELS as _L
from chuchichaestli.utils.visualization.colors import get_color, color_variant
from chuchichaestli.utils.visualization.ir import IRGraph, IRNode

__all__ = ["DiagramLevel", "ZoomSpec", "Renderer", "type_color", "type_fill"]


class DiagramLevel(IntEnum):
    """Abstraction level selected for rendering."""

    COMPONENT = 0
    LEVEL = 1
    BLOCK = 2
    LAYER = 3


def normalize_level(level: int | str | DiagramLevel) -> int:
    """Return the node-depth cutoff for a requested abstraction level.

    Args:
        level: Level as int, name, or `DiagramLevel`.
    """
    if isinstance(level, str):
        level = DiagramLevel[level.upper()]
    return int(level) + 1


@dataclass
class ZoomSpec:
    """Exemplary-zoom request for a composite diagram.

    Args:
        target: Node id to expand into its layers.
        loc: Inset location (matplotlib `inset_axes` style corner).
        size: Inset size as a fraction of the axes.
    """

    target: str
    loc: str = "upper right"
    size: float = 0.36


_TYPE_COLOR: dict[str, str] = {
    _L.CONV.value: "green",
    _L.LIN.value: "blue",
    _L.NORM.value: "orange",
    _L.DROP.value: "red",
    _L.DOWNSAMP.value: "marguerite",
    _L.UPSAMP.value: "turquoise",
    _L.ACT.value: "pink",
    _L.ATTN.value: "cyan",
    _L.EMB.value: "yellow",
    _L.RECUR.value: "purpleblue",
    _L.SHUFFLE.value: "golden",
    _L.PAD.value: "brown",
    _L.IDENT.value: "grey",
    _L.C3LI_RESBLOCK.value: "green",
    _L.C3LI_SELFATTN.value: "cyan",
    _L.C3LI_CONVATTN.value: "cyan",
    _L.C3LI_NOISEBLOCK.value: "purple",
    _L.C3LI_TIME_EMB.value: "yellow",
}

_COMPONENT_COLOR: dict[str, str] = {
    "Encoder": "blue",
    "Decoder": "green",
    "Bottleneck": "purple",
    "Latent": "purple",
    "Conditioning": "golden",
    "Network": "blue",
}


def type_color(label: str) -> str:
    """Stroke colour (hex) for a layer/block type label.

    Args:
        label: IR node `type_label`.
    """
    return get_color(_TYPE_COLOR.get(label, "grey"))


def type_fill(label: str) -> str:
    """Light fill colour (hex) for a layer/block type label.

    Args:
        label: IR node `type_label`.
    """
    return color_variant(_TYPE_COLOR.get(label, "grey"), 150)


def component_color(node: IRNode) -> str:
    """Colour name for a component node, keyed on its label.

    Args:
        node: Component IR node.
    """
    return _COMPONENT_COLOR.get(node.label, "cyan")


class Renderer(ABC):
    """Base renderer consuming a semantic `IRGraph`."""

    def __init__(self, graph: IRGraph) -> None:
        """Constructor.

        Args:
            graph: Finalized semantic IR graph.
        """
        self.graph = graph

    @abstractmethod
    def render(self) -> Any:
        """Produce the backend-native artifact (string, figure, ...)."""

    @abstractmethod
    def _supported_formats(self) -> set[str]:
        """Return the set of writable file extensions (without dot)."""

    @abstractmethod
    def _save_impl(
        self, artifact: Any, filepath: Path, width: int, height: int, scale: int
    ) -> None:
        """Write `artifact` to `filepath` (backend-specific)."""

    def save(
        self,
        filename: Path | str,
        width: int = 1920,
        height: int = 1080,
        scale: int = 4,
    ) -> Path:
        """Render and save the diagram, inferring the format from the suffix.

        Args:
            filename: Output path; the suffix selects the format.
            width: Target width in pixels.
            height: Target height in pixels.
            scale: Resolution scale factor.
        """
        filepath = Path(filename)
        fmt = filepath.suffix[1:].lower()
        if fmt not in self._supported_formats():
            raise ValueError(
                f"Unsupported format {fmt!r}; use one of "
                f"{sorted(self._supported_formats())}."
            )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._save_impl(self.render(), filepath, width, height, scale)
        return filepath

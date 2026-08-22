# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Renderer contract and shared styling for model-visualization backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args
from chuchichaestli.utils.modules import DEFAULT_MODULE_LABELS as _L
from chuchichaestli.utils.ir import IRGraph, IRNode
from chuchichaestli.utils.visualization.colors import get_color, color_variant

__all__ = ["LabelField", "TYPE_COLOR", "ZoomSpec", "Renderer", "type_color", "type_fill"]


# Node label fields shared by both backends.
LabelField = Literal["name", "channels", "kernel", "resolution", "params"]
LABEL_FIELDS = get_args(LabelField)


@dataclass
class ZoomSpec:
    """Exemplary-zoom request for a composite diagram.

    Args:
        target: Node id to expand into its layers.
        loc: Inset placement relative to the main axes. Side docks sit just
            outside the axes: `"left"`/`"right"` are full-height columns and
            `"top"`/`"bottom"` are full-width strips. The corners
            `"top-left"`/`"top-right"`/`"bottom-left"`/`"bottom-right"` hang a
            shorter column off that corner (also outside). Any value without a
            side/corner keyword (e.g. `"center"`) is a centred square inset
            inside the axes.
        size: Inset size as a fraction of the axes: the column width for
            `"left"`/`"right"` and the corners, the strip height for
            `"top"`/`"bottom"`, or the square side for a centred inset.
        bounds: Explicit `[x, y, w, h]` axes-fraction bounds; overrides
            `loc`/`size` when given (values outside `[0, 1]` place the
            inset in the figure margin).
        fields: Which fields to label each zoom layer with (comma-separated),
            any of `"name"`, `"channels"`, `"kernel"`, `"resolution"`,
            `"params"`; None shows all that apply.
    """

    target: str
    loc: str = "right"
    size: float = 0.22
    bounds: tuple[float, float, float, float] | None = None
    fields: tuple[str, ...] | None = None


TYPE_COLOR: dict[str, str] = {
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
    # Structural blocks, so color_by="type" is meaningful at the block level.
    # (Chosen among the hues that lighten to a visible fill, not near-white.)
    _L.C3LI_UNET_ENCODER.value: "green",
    _L.C3LI_UNET_DECODER.value: "turquoise",
    _L.C3LI_UNET_DOWNBLOCK.value: "green",
    _L.C3LI_UNET_UPBLOCK.value: "turquoise",
    _L.C3LI_UNET_MIDBLOCK.value: "purple",
    _L.C3LI_UNET_DOWNSAMP.value: "orange",
    _L.C3LI_UNET_UPSAMP.value: "golden",
    _L.C3LI_UNET_ATTNDOWNBLOCK.value: "pink",
    _L.C3LI_UNET_ATTNMIDBLOCK.value: "cyan",
    _L.C3LI_UNET_ATTNUPBLOCK.value: "yellow",
    _L.C3LI_UNET_GATEDATTNBLOCK.value: "cyan",
    _L.C3LI_GAN_CONVBLOCK.value: "orange",
    _L.C3LI_GAN_ATTNBLOCK.value: "cyan",
    _L.C3LI_VAE_DOWNBLOCK.value: "green",
    _L.C3LI_VAE_MIDBLOCK.value: "purple",
    _L.C3LI_VAE_UPBLOCK.value: "turquoise",
    _L.C3LI_DISCRIMINATOR.value: "brown",
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
    return get_color(TYPE_COLOR.get(label, "grey"))


def type_fill(label: str) -> str:
    """Light fill colour (hex) for a layer/block type label.

    Args:
        label: IR node `type_label`.
    """
    return color_variant(TYPE_COLOR.get(label, "grey"), 150)


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

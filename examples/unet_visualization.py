# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize a U-Net at every abstraction level, with an exemplary zoom."""

from pathlib import Path
from chuchichaestli.models.unet import UNet
from chuchichaestli.utils.visualization import (
    build_ir,
    matplotlib_diagram,
    mermaid_diagram,
    NodeRole,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
INPUT_SHAPE = (1, 1, 64, 64)


def build_model() -> UNet:
    """A small 3-level U-Net with attention on the deepest level."""
    return UNet(
        dimensions=2,
        in_channels=1,
        n_channels=32,
        out_channels=1,
        down_block_types=("DownBlock", "DownBlock", "AttnDownBlock"),
        up_block_types=("AttnUpBlock", "UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2, 4),
        skip_connection_action="concat",
    )


def main() -> None:
    """Render the U-Net across levels and save the figures to `assets/`."""
    model = build_model()
    ASSETS.mkdir(exist_ok=True)
    for level in range(4):
        matplotlib_diagram(
            model,
            level=level,
            input_shape=INPUT_SHAPE,
            color_labels=level >= 2,
            title=f"U-Net (level {level})",
        ).save(ASSETS / f"unet_level{level}.pdf")

    graph = build_ir(model, input_shape=INPUT_SHAPE)
    block = next(
        n
        for n in graph.nodes_by_role(NodeRole.BLOCK)
        if n.id.endswith("block0") and "/encoder/" in n.id
    )
    matplotlib_diagram(
        model, level=1, zoom=block.id, input_shape=INPUT_SHAPE, title="U-Net (zoom)"
    ).save(ASSETS / "unet_zoom.pdf")

    diagram = mermaid_diagram(model, input_shape=INPUT_SHAPE, group_by="encoder_decoder")
    diagram.save(ASSETS / "unet.mmd")
    print(f"Wrote U-Net figures to {ASSETS}")


if __name__ == "__main__":
    main()

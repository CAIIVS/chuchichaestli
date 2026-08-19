# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize a VAE: mirror trapezoids with a latent pinch and no skip arcs.

Same tool as the U-Net example, different topology -- the autoencoder adapter
emits an encoder/latent/decoder hierarchy without cross-level skip connections.
"""

from pathlib import Path
from chuchichaestli.models.autoencoder.vae import VAE
from chuchichaestli.utils.visualization import (
    build_ir,
    matplotlib_diagram,
    NodeRole,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
INPUT_SHAPE = (1, 3, 64, 64)


def build_model() -> VAE:
    """A small 3-level VAE."""
    return VAE(
        dimensions=2,
        in_channels=3,
        n_channels=32,
        latent_dim=4,
        out_channels=3,
        down_block_types=("AutoencoderDownBlock",) * 3,
        up_block_types=("AutoencoderUpBlock",) * 3,
        block_out_channel_mults=(1, 2, 4),
        down_layers_per_block=2,
        up_layers_per_block=2,
        encoder_mid_block_types=("AutoencoderMidBlock",),
        decoder_mid_block_types=("AutoencoderMidBlock",),
    )


def main() -> None:
    """Render the VAE across levels and save the figures to `assets/`."""
    model = build_model()
    ASSETS.mkdir(exist_ok=True)
    for level in range(4):
        matplotlib_diagram(
            model, level=level, input_shape=INPUT_SHAPE, title=f"VAE (level {level})"
        ).save(ASSETS / f"vae_level{level}.pdf")

    graph = build_ir(model, input_shape=INPUT_SHAPE)
    block = next(
        n
        for n in graph.nodes_by_role(NodeRole.BLOCK)
        if n.id.endswith("block0") and "/encoder/" in n.id
    )
    matplotlib_diagram(
        model, level=1, zoom=block.id, input_shape=INPUT_SHAPE, title="VAE (zoom)"
    ).save(ASSETS / "vae_zoom.pdf")
    print(f"Wrote VAE figures to {ASSETS}")


if __name__ == "__main__":
    main()

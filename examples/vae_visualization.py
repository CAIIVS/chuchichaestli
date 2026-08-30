# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize a VAE diagram including encoder, latent, and decoder.

The matplotlib backend renders the high-level components, blocks, or layers,
including the latent space rendered as an hourglass node.
The mermaid backend generates a Mermaid file to be visualized in the browser.

Usage (requires `chuchichaestli[viz]`):
```python
    [uv run --extra viz] python examples/vae_visualization.py
```
"""

from pathlib import Path
from chuchichaestli.utils.info import summary
from chuchichaestli.utils.visualization import matplotlib_diagram

DIR = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 3, 64, 64)

# Build a simple VAE model
# --8<-- [start:vae]
from chuchichaestli.models.autoencoder import VAE, VAEDecoder, VAEEncoder

encoder = VAEEncoder(
    dimensions=2,
    in_channels=3,
    n_channels=32,
    out_channels=4,
    down_block_types=("AutoencoderDownBlock",) * 3,
    block_out_channel_mults=(1, 2, 4),
    num_layers_per_block=2,
    mid_block_types=("AutoencoderMidBlock",),
)
decoder = VAEDecoder(
    dimensions=2,
    in_channels=4,
    n_channels=encoder.bottleneck_channels,
    out_channels=3,
    up_block_types=("AutoencoderUpBlock",) * 3,
    block_out_channel_mults=(1, 2, 4),
    num_layers_per_block=2,
    mid_block_types=("AutoencoderMidBlock",),
)
model = VAE(encoder, decoder)
# --8<-- [end:vae]

print(
    summary(
        model,
        input_shape=INPUT_SHAPE,
        depth=3,
        columns=("input", "output", "kernel", "params", "mult_adds"),
    )
)

# Create matplotlib graph diagrams for several levels
for level in range(4):
    matplotlib_diagram(
        model,
        level=level,
        input_shape=INPUT_SHAPE,
        node_size="large",
        label_fields=("channels", "resolution", "params"),
        color_by="name",
        title=f"VAE (level {level})",
    ).save(DIR / f"vae_level{level}.svg")
matplotlib_diagram(
    model,
    level=2,
    input_shape=INPUT_SHAPE,
    label_fields=("channels", "resolution", "params"),
    color_by="name",
    zoom="model/encoder/level0/block0",
    zoom_loc="bottom",
    zoom_size=0.15,
    title="VAE (zoom)"
).save(DIR / "vae_zoom.svg")
print(f"Wrote VAE figures to {DIR}")

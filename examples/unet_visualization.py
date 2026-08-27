# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize a U-Net diagram at every abstraction level.

The matplotlib backend renders the high-level components, blocks, or layers,
including shortcuts within the blocks and skip connections between U-Net levels.
The mermaid backend generates a Mermaid file to be visualized in the browser.

Usage (requires `chuchichaestli[viz]`):
```python
    [uv run --extra viz] python examples/unet_visualization.py
```
"""

# Build a simple U-Net with Attention blocks near the bottleneck
# --8<-- [start:unet]
from chuchichaestli.models.unet import UNet

model = UNet(
    dimensions=2,        # spatial dimensions
    in_channels=1,       # input image channels such as RGB or monochrome
    n_channels=32,       # channels of first hidden layer
    out_channels=1,      # output image channels such as RGB or monochrome
    down_block_types=("DownBlock", "DownBlock", "AttnDownBlock"),  # residual blocks and an attention block in the lowest level
    up_block_types=("AttnUpBlock", "UpBlock", "UpBlock"),  # an attention block in the lowest level and residual blocks
    block_out_channel_mults=(1, 2, 4),  # channel multipliers with each level
    res_act_fn="silu",   # SiLU activations
    res_dropout=0.4,     # dropout for residual blocks
    attn_n_heads=1,      # number of attention heads per block,
    skip_connection_action="concat",    # skip connections are concatenated in decoder
)
# --8<-- [end:unet]

# Inspect model graph summary on stdout
# --8<-- [start:summary]
from chuchichaestli.utils.info import summary

INPUT_SHAPE = (1, 1, 64, 64)
print(
    summary(
        model,
        input_shape=INPUT_SHAPE,
        depth=3,
        columns=("input", "output", "kernel", "params", "mult_adds"),
    )
)
# --8<-- [end:summary]

# Create matplotlib graph diagrams for several levels
# --8<-- [start:matplotlib]
from pathlib import Path
from chuchichaestli.utils.visualization import (
    matplotlib_diagram,
    mermaid_diagram,
    ZoomSpec,
)

DIR = Path(__file__).resolve().parent
LEVEL_LABELS = ("channels", "resolution", "params")

for level in range(4):
    matplotlib_diagram(
        model,
        level=level,
        input_shape=INPUT_SHAPE,
        node_size="small" if level >= 3 else "medium",
        label_fields=("name",) + LEVEL_LABELS if level == 1 else LEVEL_LABELS,
        color_by="component" if level <= 1 else "type",
        title=f"U-Net (level {level})",
    ).save(DIR / f"unet_level{level}.svg")
# --8<-- [end:matplotlib]

# `zoom=True` auto-picks a representative block; pass a node id (or `ZoomSpec`)
# to choose another, with `zoom_loc`/`zoom_size`/etc to place the inset.
# Pass a list of `ZoomSpec`s to draw multiple zooms, each with its own settings.
# --8<-- [start:zoom]
from chuchichaestli.utils.visualization import ZoomSpec

matplotlib_diagram(
    model,
    level=2,
    zoom=[
        ZoomSpec("model/encoder/level0/block0", loc="left", size=0.15),
        ZoomSpec("model/decoder/level2/block4", loc="right", size=0.2),
    ],
    input_shape=INPUT_SHAPE,
    label_fields=("channels", "resolution", "params"),
    color_by="type",
    title="U-Net (zoom)",
).save(DIR / "unet_zoom.svg")
# --8<-- [end:zoom]

# Create a mermaid graph diagram
# --8<-- [start:mermaid]
from chuchichaestli.utils.visualization import mermaid_diagram

diagram = mermaid_diagram(
    model, input_shape=INPUT_SHAPE, group_by=["encoder_decoder", "block"], max_depth=4
)

diagram.save(DIR / "unet.mmd")
# --8<-- [end:mermaid]

print(f"Wrote U-Net figures to {DIR}")

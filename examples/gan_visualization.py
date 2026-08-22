# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize a PatchGAN discriminator diagram.

The matplotlib backend renders the diagram as sequential node chain for higher
levels, and would collapse to a single component graph for levels lower than 2.
The mermaid backend generates a Mermaid file to be visualized in the browser.
"""

from pathlib import Path
from chuchichaestli.models.adversarial.discriminator import PatchDiscriminator
from chuchichaestli.utils.visualization import matplotlib_diagram, mermaid_diagram, summary

DIR = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 3, 128, 128)

# Build a simple PatchGAN discriminator model
model = PatchDiscriminator(dimensions=2, in_channels=3, n_channels=64)
print(
    summary(
        model,
        input_shape=INPUT_SHAPE,
        depth=3,
        columns=("input", "output", "kernel", "params", "mult_adds"),
    )
)

# Create matplotlib graph diagrams for several levels
for level in range(2, 4):
    matplotlib_diagram(
        model,
        level=level,
        input_shape=INPUT_SHAPE,
        node_size="large",
        label_fields=("name", "channels", "resolution", "params"),
        color_by="name" if level==2 else "type",
        title=f"PatchGAN (level {level})",
    ).save(DIR / f"gan_level{level}.svg")

# Create a mermaid graph diagram
mermaid_diagram(model, input_shape=INPUT_SHAPE).save(DIR / "gan.mmd")
print(f"Wrote PatchGAN figures to {DIR}")

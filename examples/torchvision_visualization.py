# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize torchvision pretrained models (a ResNet and a ViT).

External models have no dedicated adapter, so the generic fallback builds the
IR from data tracing: the diagrams are denser and less structured
than the chuchichaestli models, but summaries and figures still render.

Usage (requires `chuchichaestli[viz]`):
```python
    [uv run --extra viz] python examples/torchvision_visualization.py
```
"""

from pathlib import Path
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)
from chuchichaestli.utils.visualization import matplotlib_diagram, mermaid_diagram, summary

DIR = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 3, 224, 224)
LEVEL = 4

def load(ctor, weights):
    """Load a pretrained model, falling back to random init if offline."""
    try:
        return ctor(weights=weights)
    except Exception:  # noqa: BLE001 - network/cache issues shouldn't break the demo
        print(f"  (could not fetch weights for {ctor.__name__}; using random init)")
        return ctor(weights=None)


models = {
    "resnet18": load(resnet18, ResNet18_Weights.DEFAULT),
    "vit_b_16": load(vit_b_16, ViT_B_16_Weights.DEFAULT),
}

for name, model in models.items():
    model.eval()
    print(
        summary(
            model,
            input_shape=INPUT_SHAPE,
            depth=LEVEL,
            columns=("input", "output", "params", "mult_adds"),
        )
    )
    matplotlib_diagram(
        model,
        level=LEVEL,
        input_shape=INPUT_SHAPE,
        node_size="small",
        label_fields=("channels", "kernel", "resolution", "params"),
        color_by="name",
        title=f"{name} (level {LEVEL})",
    ).save(DIR / f"{name}.svg")
    # Mermaid groups the many nodes by layer type for a browsable overview.
    mermaid_diagram(model, input_shape=INPUT_SHAPE, group_by="type").save(DIR / f"{name}.mmd")

print(f"Wrote torchvision figures to {DIR}")

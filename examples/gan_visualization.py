# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visualize a PatchGAN discriminator via the generic sequential fallback.

The discriminator has no encoder/decoder or skip semantics, so the generic
adapter degrades gracefully to a single-level block chain.
"""

from pathlib import Path
from chuchichaestli.models.adversarial.discriminator import PatchDiscriminator
from chuchichaestli.utils.visualization import matplotlib_diagram, mermaid_diagram

ASSETS = Path(__file__).resolve().parent.parent / "assets"
INPUT_SHAPE = (1, 3, 128, 128)


def build_model() -> PatchDiscriminator:
    """A 70x70 PatchGAN discriminator."""
    return PatchDiscriminator(dimensions=2, in_channels=3, n_channels=64)


def main() -> None:
    """Render the discriminator and save the figures to `assets/`."""
    model = build_model()
    ASSETS.mkdir(exist_ok=True)
    for level in (0, 1, 2):
        matplotlib_diagram(
            model, level=level, input_shape=INPUT_SHAPE, title=f"PatchGAN (level {level})"
        ).save(ASSETS / f"gan_level{level}.pdf")
    mermaid_diagram(model, input_shape=INPUT_SHAPE).save(ASSETS / "gan.mmd")
    print(f"Wrote PatchGAN figures to {ASSETS}")


if __name__ == "__main__":
    main()

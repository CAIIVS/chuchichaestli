# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the visualization renderers."""

import subprocess
import sys
import pytest
from chuchichaestli.models.unet import UNet
from chuchichaestli.utils.visualization import matplotlib_diagram, mermaid_diagram

pytest.importorskip("matplotlib")


def _unet():
    return UNet(
        dimensions=2,
        in_channels=1,
        n_channels=32,
        out_channels=1,
        down_block_types=("DownBlock", "AttnDownBlock"),
        up_block_types=("AttnUpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2),
        skip_connection_action="concat",
    )


def test_package_import_does_not_load_matplotlib():
    """Importing the visualization package must not import matplotlib."""
    code = (
        "import sys, chuchichaestli.utils.visualization as v; "
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_render_each_level_saves(level, tmp_path):
    """Every abstraction level renders and saves a non-empty file."""
    diagram = matplotlib_diagram(_unet(), level=level, input_shape=(1, 1, 64, 64))
    for ext in ("pdf", "svg", "png"):
        path = diagram.save(tmp_path / f"unet_l{level}.{ext}")
        assert path.exists() and path.stat().st_size > 0


def test_zoom_produces_inset():
    """A composite zoom renders a main axes plus an inset axes."""
    from chuchichaestli.utils.visualization import build_ir, NodeRole

    graph = build_ir(_unet(), input_shape=(1, 1, 64, 64))
    block = next(n for n in graph.nodes_by_role(NodeRole.BLOCK) if "/encoder/" in n.id)
    fig = matplotlib_diagram(
        _unet(), level=1, zoom=block.id, input_shape=(1, 1, 64, 64)
    ).render()
    assert any(ax.child_axes for ax in fig.axes)


def test_unknown_zoom_target_raises():
    """An unknown zoom target raises ValueError."""
    with pytest.raises(ValueError):
        matplotlib_diagram(_unet(), zoom="model/nope", input_shape=(1, 1, 64, 64))


def test_unsupported_format_raises(tmp_path):
    """Saving to an unsupported extension raises ValueError."""
    diagram = matplotlib_diagram(_unet(), input_shape=(1, 1, 64, 64))
    with pytest.raises(ValueError):
        diagram.save(tmp_path / "unet.gif")


def test_mermaid_renders_skip_edges():
    """The mermaid backend renders dashed skip edges for a U-Net."""
    assert "-.->" in str(mermaid_diagram(_unet(), input_shape=(1, 1, 64, 64)))

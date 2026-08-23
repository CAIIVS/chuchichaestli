# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the visualization renderers."""

import os
import subprocess
import sys
import pytest
from chuchichaestli.models.unet import UNet
from chuchichaestli.utils.visualization import (
    matplotlib_diagram,
    mermaid_diagram,
    build_ir,
    ZoomSpec,
    NodeRole,
)

pytest.importorskip("matplotlib")
from chuchichaestli.utils.visualization.mpl import MatplotlibRenderer  # noqa: E402


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


@pytest.mark.parametrize("max_depth", [None, 1, 2, 3, 4])
def test_mermaid_renders_skip_edges(max_depth):
    """The mermaid backend renders skip edges at every depth, layers included."""
    text = str(
        mermaid_diagram(_unet(), input_shape=(1, 1, 64, 64), max_depth=max_depth)
    )
    # Match the label: residual edges use the same dashed arrow.
    assert "-.->|skip|" in text


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_matplotlib_draws_skip_arcs(level, monkeypatch):
    """Every U-Net skip gets an arc, also at layer level where blocks vanish."""
    arcs = []
    original = MatplotlibRenderer._arc

    def spy(self, mpl, ax, src, tgt, label, height):
        arcs.append(label)
        return original(self, mpl, ax, src, tgt, label, height)

    monkeypatch.setattr(MatplotlibRenderer, "_arc", spy)
    matplotlib_diagram(_unet(), level=level, input_shape=(1, 1, 64, 64)).render()
    assert arcs


@pytest.mark.parametrize("color_by", ["auto", "component", "type", "name"])
def test_color_by_modes_render(color_by, tmp_path):
    """Every color_by mode renders a non-empty figure."""
    diagram = matplotlib_diagram(
        _unet(), level=2, color_by=color_by, input_shape=(1, 1, 64, 64)
    )
    assert diagram.save(tmp_path / f"c_{color_by}.png").stat().st_size > 0


def test_unknown_color_by_raises():
    """An unknown color_by raises ValueError."""
    with pytest.raises(ValueError):
        matplotlib_diagram(_unet(), color_by="rainbow", input_shape=(1, 1, 64, 64))


def test_label_fields_render(tmp_path):
    """All label fields (incl. kernel) render at the layer level."""
    diagram = matplotlib_diagram(
        _unet(),
        level=3,
        label_fields=("name", "channels", "kernel", "resolution", "params"),
        input_shape=(1, 1, 64, 64),
    )
    assert diagram.save(tmp_path / "lf.png").stat().st_size > 0


def test_unknown_label_field_raises():
    """An unknown label field raises ValueError."""
    with pytest.raises(ValueError):
        matplotlib_diagram(_unet(), label_fields=("bogus",), input_shape=(1, 1, 64, 64))


def test_node_size_scales_and_validates():
    """node_size scales the node width; an unknown value raises."""
    graph = build_ir(_unet(), input_shape=(1, 1, 64, 64))
    slots = [MatplotlibRenderer(graph, node_size=s)._slot for s in ("small", "medium", "large")]
    assert slots[0] < slots[1] < slots[2]
    with pytest.raises(ValueError):
        MatplotlibRenderer(graph, node_size="huge")


def test_zoom_locations_have_unique_bounds():
    """Each zoom location (edges, corners, center) maps to a distinct inset box."""
    graph = build_ir(_unet(), input_shape=(1, 1, 64, 64))
    target = graph.nodes_by_role(NodeRole.BLOCK)[0].id
    locs = [
        "left", "right", "top", "bottom", "center",
        "top-left", "top-right", "bottom-left", "bottom-right",
    ]
    bounds = {
        tuple(
            MatplotlibRenderer(
                graph, zoom=ZoomSpec(target, loc=loc, size=0.2)
            )._inset_bounds(ZoomSpec(target, loc=loc, size=0.2))
        )
        for loc in locs
    }
    assert len(bounds) == len(locs)


def test_multi_zoom_draws_multiple_insets():
    """A list of zoom targets draws one inset per target."""
    graph = build_ir(_unet(), input_shape=(1, 1, 64, 64))
    blocks = [n.id for n in graph.nodes_by_role(NodeRole.BLOCK)]
    fig = matplotlib_diagram(
        _unet(),
        level=1,
        input_shape=(1, 1, 64, 64),
        zoom=[ZoomSpec(blocks[0], loc="left"), ZoomSpec(blocks[-1], loc="right")],
    ).render()
    n_insets = sum(len(ax.child_axes) for ax in fig.axes)
    assert n_insets >= 2


def test_mermaid_mmd_export_is_utf8(tmp_path):
    """A `.mmd` export writes UTF-8 even where the locale is ASCII-only."""
    out = tmp_path / "vae.mmd"
    code = (
        "from chuchichaestli.models.autoencoder.vae import VAE;"
        "from chuchichaestli.utils.visualization import mermaid_diagram;"
        "m = VAE(dimensions=2, in_channels=3, n_channels=16, latent_dim=4,"
        " out_channels=3, down_block_types=('AutoencoderDownBlock',),"
        " up_block_types=('AutoencoderUpBlock',), block_out_channel_mults=(1,),"
        " down_layers_per_block=1, up_layers_per_block=1,"
        " encoder_mid_block_types=('AutoencoderMidBlock',),"
        " decoder_mid_block_types=('AutoencoderMidBlock',));"
        "mermaid_diagram(m, input_shape=(1, 3, 32, 32), max_depth=3)"
        f".save(r'{out}')"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    # The VAE's reparameterization label carries non-ASCII glyphs.
    assert "\u03bc" in out.read_text(encoding="utf-8")

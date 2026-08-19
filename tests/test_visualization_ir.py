# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the visualization semantic IR and adapters."""

import pytest
from chuchichaestli.models.unet import UNet
from chuchichaestli.models.autoencoder.vae import VAE
from chuchichaestli.models.autoencoder.vqvae import VQVAE
from chuchichaestli.models.adversarial.discriminator import PatchDiscriminator
from chuchichaestli.utils.visualization.build import build_ir
from chuchichaestli.utils.visualization.ir import NodeRole, EdgeKind


def _unet(dimensions=2, num_blocks_per_level=1, skip_connection_action="concat"):
    return UNet(
        dimensions=dimensions,
        in_channels=1,
        n_channels=32,
        out_channels=1,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2),
        num_blocks_per_level=num_blocks_per_level,
        skip_connection_action=skip_connection_action,
    )


def _vae(cls=VAE):
    return cls(
        dimensions=2,
        in_channels=3,
        n_channels=16,
        latent_dim=4,
        out_channels=3,
        down_block_types=("AutoencoderDownBlock", "AutoencoderDownBlock"),
        up_block_types=("AutoencoderUpBlock", "AutoencoderUpBlock"),
        block_out_channel_mults=(1, 2),
        down_layers_per_block=1,
        up_layers_per_block=1,
        encoder_mid_block_types=("AutoencoderMidBlock",),
        decoder_mid_block_types=("AutoencoderMidBlock",),
        res_groups=4,
        encoder_groups=4,
        decoder_groups=4,
        attn_groups=4,
    )


def _shape(dimensions):
    return (1, 1) + (16,) * dimensions


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_unet_components(dimensions):
    """U-Net yields encoder/bottleneck/decoder with matching level counts."""
    graph = build_ir(_unet(dimensions), input_shape=_shape(dimensions))
    labels = [n.label for n in graph.nodes_by_role(NodeRole.COMPONENT)]
    assert labels == ["Encoder", "Bottleneck", "Decoder"]
    encoder = graph.node("model/encoder")
    decoder = graph.node("model/decoder")
    n_levels = 2
    assert sum(c.role == NodeRole.LEVEL for c in encoder.children) == n_levels
    assert sum(c.role == NodeRole.LEVEL for c in decoder.children) == n_levels


@pytest.mark.parametrize("num_blocks_per_level", [1, 2])
def test_unet_skip_pairing(num_blocks_per_level):
    """Skip edges mirror encoder and decoder levels (LIFO pairing)."""
    graph = build_ir(
        _unet(num_blocks_per_level=num_blocks_per_level), input_shape=_shape(2)
    )
    skips = [e for e in graph.edges if e.kind == EdgeKind.SKIP]
    assert len(skips) == 2
    n_levels = 2
    for edge in skips:
        source = graph.node(edge.source_id)
        target = graph.node(edge.target_id)
        assert source.id.startswith("model/encoder/")
        assert target.id.startswith("model/decoder/")
        assert (
            source.geometry.level_index + target.geometry.level_index == n_levels - 1
        )


def test_unet_no_skip_when_action_none():
    """No skip edges are emitted when skip_connection_action is None."""
    graph = build_ir(_unet(skip_connection_action=None), input_shape=_shape(2))
    assert not [e for e in graph.edges if e.kind == EdgeKind.SKIP]


@pytest.mark.parametrize("cls,kind", [(VAE, "reparameterization"), (VQVAE, "codebook")])
def test_autoencoder_latent(cls, kind):
    """Autoencoders expose encoder/latent/decoder, zero skips, typed latent."""
    model = _vae(cls)
    graph = build_ir(model, input_shape=(1, 3, 32, 32))
    labels = [n.label for n in graph.nodes_by_role(NodeRole.COMPONENT)]
    assert labels == ["Encoder", "Latent", "Decoder"]
    assert not [e for e in graph.edges if e.kind == EdgeKind.SKIP]
    latent = graph.node("model/latent")
    assert latent.meta["latent_kind"] == kind
    encoder_levels = sum(
        c.role == NodeRole.LEVEL for c in graph.node("model/encoder").children
    )
    assert encoder_levels == model.encoder.levels


def test_view_preserves_params_and_collapses():
    """Collapsing to a shallower depth preserves total params, prunes depth."""
    graph = build_ir(_unet(), input_shape=_shape(2))
    total = graph.root.num_params
    for depth in range(5):
        view = graph.view(depth)
        assert view.root.num_params == total
        assert max(n.depth for n in view.root.walk()) <= depth


def test_zoom_expands_only_target():
    """Zoom expands one block's subtree while collapsing the rest."""
    graph = build_ir(_unet(), input_shape=_shape(2))
    target = graph.nodes_by_role(NodeRole.BLOCK)[0]
    zoomed = graph.zoom(target.id, depth=1)
    assert zoomed.node(target.id).children  # target expanded to layers
    # a sibling component that is not on the zoom path stays collapsed
    decoder = zoomed.node("model/decoder")
    assert not decoder.children


def test_zoom_unknown_target_raises():
    """Zooming an unknown node id raises ValueError."""
    graph = build_ir(_unet(), input_shape=_shape(2))
    with pytest.raises(ValueError):
        graph.zoom("model/nonexistent")


def test_generic_adapter_on_discriminator():
    """A discriminator builds without skips and without crashing."""
    model = PatchDiscriminator(dimensions=2, in_channels=3, n_channels=16)
    graph = build_ir(model, input_shape=(1, 3, 64, 64))
    assert graph.nodes_by_role(NodeRole.BLOCK)
    assert not [e for e in graph.edges if e.kind == EdgeKind.SKIP]

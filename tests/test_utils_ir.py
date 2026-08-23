# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the visualization semantic IR and adapters."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.unet import UNet
from chuchichaestli.models.autoencoder.vae import VAE
from chuchichaestli.models.autoencoder.vqvae import VQVAE
from chuchichaestli.models.adversarial.discriminator import PatchDiscriminator
from chuchichaestli.utils.visualization import mermaid_diagram
from chuchichaestli.utils.ir import NodeRole, EdgeKind, build_ir, normalize_level
from chuchichaestli.utils.info import summary


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


def _true_params(model):
    return sum(p.numel() for p in model.parameters())


class _DirectParamBlock(nn.Module):
    """A non-leaf module that also holds a parameter directly (like attention)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))  # 16 params, not in a submodule
        self.proj = nn.Linear(4, 4)  # 20 params in a submodule

    def forward(self, x):
        return self.proj(x @ self.weight)


class _Irregular(nn.Module):
    """A model whose branches bottom out at different depths."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)  # leaf at depth 1
        self.b = nn.Sequential(  # deeper branch
            nn.Conv2d(4, 4, 3, padding=1), nn.Sequential(nn.Conv2d(4, 4, 1))
        )

    def forward(self, x):
        return self.b(self.a(x))


@pytest.mark.parametrize(
    "factory,shape",
    [(lambda: _unet(), _shape(2)), (lambda: _vae(VAE), (1, 3, 32, 32))],
)
def test_param_count_matches_model(factory, shape):
    """Aggregated IR params equal the model's true parameter count."""
    model = factory()
    assert build_ir(model, input_shape=shape).root.num_params == _true_params(model)


def test_param_count_includes_direct_params():
    """Params a non-leaf module holds directly are counted, not just leaves."""
    model = _DirectParamBlock()
    graph = build_ir(model, input_shape=(1, 4))
    assert graph.root.num_params == _true_params(model) == 36


def test_generic_adapter_no_duplicate_nodes():
    """The generic adapter mirrors the module tree without duplicating nodes."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    graph = build_ir(model, input_shape=(1, 4))
    layers = [n for n in graph.root.walk() if n.role == NodeRole.LAYER]
    assert len(layers) == 3  # one node per leaf module, no over-linking
    assert graph.root.num_params == _true_params(model)


class _ResidualBranch(nn.Module):
    """A branched block: a main conv path plus a parallel shortcut branch."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.shortcut = nn.Conv2d(8, 8, 1, bias=False)  # fed from the block input

    def forward(self, x):
        x = self.stem(x)
        return self.bn1(self.conv1(x)) + self.shortcut(x)


def test_generic_adapter_traces_branched_dataflow():
    """Forward edges follow real tensor dataflow, not registration order."""
    graph = build_ir(_ResidualBranch(), input_shape=(1, 3, 8, 8))
    fwd = {(e.source_id, e.target_id) for e in graph.edges if e.kind == EdgeKind.FORWARD}

    def nid(suffix):
        return next(
            n.id
            for n in graph.root.walk()
            if n.role == NodeRole.LAYER and n.id.endswith(suffix)
        )

    stem, conv1, bn1, shortcut = nid("/stem"), nid("/conv1"), nid("/bn1"), nid("/shortcut")
    # Both the main path and the shortcut branch off the block input (stem).
    assert (stem, conv1) in fwd
    assert (stem, shortcut) in fwd
    # No fabricated serial edge from the main path into the parallel shortcut.
    assert (bn1, shortcut) not in fwd
    # The graph stays connected: only the entry conv has no producer.
    leaves = [n for n in graph.root.walk() if n.role == NodeRole.LAYER]
    targets = {t for _, t in fwd}
    assert [n.id for n in leaves if n.id not in targets] == [stem]


def test_generic_adapter_falls_back_to_chain_without_trace():
    """Structure-only builds (no input) keep a best-effort sequential chain."""
    graph = build_ir(_ResidualBranch())  # no input_shape -> no dataflow trace
    fwd = [(e.source_id, e.target_id) for e in graph.edges if e.kind == EdgeKind.FORWARD]
    leaves = [n for n in graph.root.walk() if n.role == NodeRole.LAYER]
    assert len(fwd) == len(leaves) - 1  # a single serial chain over the leaves


def test_representative_prefers_multilayer_block():
    """`representative` skips single-layer heads for a real multi-layer block."""
    rep = build_ir(_unet(), input_shape=_shape(2)).representative(NodeRole.BLOCK)
    assert len(rep.children) > 1


def test_view_frontier_is_monotonic_for_irregular_model():
    """Deeper levels stay finer: the drawn frontier never shrinks."""
    graph = build_ir(_Irregular(), input_shape=(1, 3, 8, 8))
    counts = [
        sum(1 for n in graph.view(normalize_level(lvl)).root.walk() if not n.children)
        for lvl in range(4)
    ]
    assert counts == sorted(counts)


def test_summary_reports_true_totals():
    """The summary footer reports the model's true total and trainable params."""
    model = _unet()
    text = summary(model, input_shape=_shape(2))
    assert f"Total params: {_true_params(model):,}" in text
    assert f"Trainable params: {_true_params(model):,}" in text


def test_summary_counts_direct_and_frozen_params():
    """Trainable/non-trainable are correct with direct params and frozen weights."""
    model = _DirectParamBlock()  # 16 direct + 20 submodule params
    assert "Trainable params: 36" in summary(model, input_shape=(1, 4))
    for p in model.parameters():
        p.requires_grad_(False)
    text = summary(model, input_shape=(1, 4))
    assert "Trainable params: 0" in text
    assert "Non-trainable params: 36" in text


def test_summary_columns_and_depth():
    """Requested columns appear and a shallower depth lists fewer rows."""
    model = _unet()
    text = summary(
        model, input_shape=_shape(2), depth=1, columns=("kernel", "params", "mult_adds")
    )
    assert "Kernel" in text and "Mult-Adds" in text and "Total mult-adds" in text
    shallow = len(summary(model, input_shape=_shape(2), depth=0).splitlines())
    deep = len(summary(model, input_shape=_shape(2), depth=3).splitlines())
    assert shallow < deep


def test_summary_unknown_column_raises():
    """An unknown summary column raises ValueError."""
    with pytest.raises(ValueError):
        summary(_unet(), input_shape=_shape(2), columns=("bogus",))


def test_mermaid_label_fields_map_to_flags():
    """label_fields drive the mermaid show_* flags."""
    full = mermaid_diagram(_unet(), input_shape=_shape(2), label_fields=("name", "params", "resolution"))
    assert full.show_names and full.show_params and full.show_shapes
    minimal = mermaid_diagram(_unet(), input_shape=_shape(2), label_fields=("name",))
    assert minimal.show_names and not minimal.show_params and not minimal.show_shapes


def test_mermaid_unknown_label_field_raises():
    """An unknown mermaid label field raises ValueError."""
    with pytest.raises(ValueError):
        mermaid_diagram(_unet(), input_shape=_shape(2), label_fields=("bogus",))


def test_mermaid_block_grouping():
    """group_by='block' titles subgraphs by the parent block class name."""
    diagram = mermaid_diagram(_unet(), input_shape=_shape(2), max_depth=4, group_by="block")
    assert any(v in ("DownBlock", "UpBlock", "MidBlock") for v in diagram._group_labels.values())


def test_mermaid_nested_group_by():
    """A list group_by nests block subgraphs inside encoder/decoder subgraphs."""
    text = mermaid_diagram(
        _unet(), input_shape=_shape(2), max_depth=4, group_by=["encoder_decoder", "block"]
    ).generate()
    assert 'subgraph Encoder["Encoder"]' in text
    assert text.count("subgraph") > 3


class _BatchNormModel(nn.Module):
    """Minimal model carrying BatchNorm running statistics."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        """Convolve, normalize, and drop only while training."""
        x = self.norm(self.conv(x))
        if self.training:
            x = self.drop(x)
        return x


def test_build_ir_does_not_mutate_running_stats():
    """Building the IR of a training-mode model leaves its buffers untouched."""
    model = _BatchNormModel()
    model.train()
    running_mean = model.norm.running_mean.clone()
    build_ir(model, input_shape=(1, 1, 16, 16))
    assert torch.equal(running_mean, model.norm.running_mean)
    assert model.norm.num_batches_tracked.item() == 0
    assert model.training


def test_mermaid_label_newlines_become_line_breaks():
    """A label carrying a newline stays one mermaid declaration per line."""
    text = mermaid_diagram(_vae(VAE), input_shape=(1, 3, 32, 32), max_depth=3).generate()
    assert "Reparameterize</br>(" in text
    assert all(line.count('"') % 2 == 0 for line in text.splitlines())

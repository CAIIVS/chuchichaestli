# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standard autoencoder tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.autoencoder import Autoencoder, Decoder, Encoder


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels",
    [
        (1, 1, 64, 4, 1),
        (2, 1, 64, 4, 1),
        (3, 1, 64, 4, 1),
        (1, 1, 32, 8, 1),
        (2, 1, 32, 8, 1),
        (3, 1, 32, 8, 1),
        (1, 1, 8, 4, 3),
        (2, 1, 8, 4, 3),
        (3, 1, 8, 4, 3),
    ],
)
def test_autoencoder_init(
    dimensions, in_channels, n_channels, latent_dim, out_channels
):
    """Test the Autoencoder module initialization."""
    model = Autoencoder.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels},
        latent_proj=True,
        latent_deproj=True,
    )
    assert isinstance(model.encoder, nn.Module)
    assert isinstance(model.decoder, nn.Module)
    assert hasattr(model, "latent_proj")
    assert hasattr(model, "latent_deproj")


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (1, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_autoencoder_blocks(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the Autoencoder module."""
    model = Autoencoder.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert isinstance(out, torch.Tensor)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (1, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_autoencoder_latent_dim(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the Autoencoder module (latent dim)."""
    model = Autoencoder.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    spatial_dims = (wh // model.f_comp,) * dimensions
    assert model.levels == (len(down_block_types), len(up_block_types))
    assert model.f_comp == 2 ** (len(down_block_types) - 1)
    assert model.f_exp == 2 ** (len(up_block_types) - 1)
    assert model.compute_latent_shape(shape) == (1, latent_dim, *spatial_dims)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (1, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_autoencoder_forward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the VAE module (forward pass)."""
    model = Autoencoder.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (
        1,
        out_channels,
        *(model.f_exp / model.f_comp * wh,) * dimensions,
    )


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (1, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 8, 1, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 3, 16, 8, 3, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 3, 16, 8, 3, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 3, 16, 8, 3, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 4, 1, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 4, 1, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 4, 1, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vae_backward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the Autoencoder module (backward pass)."""
    model = Autoencoder.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 32
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    loss = nn.functional.mse_loss(out, sample)
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


def test_autoencoder_inspect():
    """Test the Autoencoder module inspection."""
    model = Autoencoder.build(
        dimensions=2,
        in_channels=1,
        latent_dim=4,
        out_channels=1,
        encoder_args={"n_channels": 64},
    )
    try:
        from torchinfo import summary

        summary(
            model,
            (1, 1, 256, 256),
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=7,
        )
    except ImportError:
        print(model)
    print()


if __name__ == "__main__":
    pytest.main(["-v", "test_autoencoder.py"])


PER_COMPONENT_CONF = {
    "latent_dim": 4,
    "res_groups": 4,
}
PER_COMPONENT_ENC = {
    "n_channels": 16,
    "block_out_channel_mults": (1, 2, 2),
    "down_block_types": ("AutoencoderDownBlock",) * 3,
    "mid_block_types": ("AutoencoderMidBlock",),
    "num_layers_per_block": 1,
}
PER_COMPONENT_DEC = {
    "block_out_channel_mults": (1, 2, 2),
    "up_block_types": ("AutoencoderUpBlock",) * 3,
    "mid_block_types": ("AutoencoderMidBlock",),
    "num_layers_per_block": 1,
}


def test_per_component_args_follow_each_component_own_path():
    """Test that each component reads its override dict in its own data-flow order."""
    model = Autoencoder.build(
        **PER_COMPONENT_CONF,
        encoder_args={
            **PER_COMPONENT_ENC,
            "res_args": {"res_dropout": (0.1, 0.2, 0.4, 0.5)},
        },
        decoder_args={
            **PER_COMPONENT_DEC,
            "res_args": {"res_dropout": (0.5, 0.4, 0.2, 0.1)},
        },
    )
    encoder_levels = [s[0].res_block.dropout.p for s in model.encoder.down_blocks[::2]]
    decoder_levels = [s[0].res_block.dropout.p for s in model.decoder.up_blocks[::2]]
    assert encoder_levels == [0.1, 0.2, 0.4]
    assert [b.res_block.dropout.p for b in model.encoder.mid_blocks] == [0.5]
    assert [b.res_block.dropout.p for b in model.decoder.mid_blocks] == [0.5]
    assert decoder_levels == [0.4, 0.2, 0.1]


def test_shared_scalars_fill_keys_absent_from_the_override():
    """Test that an override dict replaces only the keys it names."""
    model = Autoencoder.build(
        **PER_COMPONENT_CONF,
        res_dropout=0.3,
        encoder_args={**PER_COMPONENT_ENC, "res_args": {"res_kernel_size": 1}},
        decoder_args=PER_COMPONENT_DEC,
    )
    assert model.encoder.down_blocks[0][0].res_block.dropout.p == 0.3


def test_each_component_gets_an_independent_argument_dict():
    """Test that an encoder override does not leak into the decoder."""
    model = Autoencoder.build(
        **PER_COMPONENT_CONF,
        res_dropout=0.3,
        encoder_args={**PER_COMPONENT_ENC, "res_args": {"res_dropout": 0.6}},
        decoder_args=PER_COMPONENT_DEC,
    )
    assert model.encoder.down_blocks[0][0].res_block.dropout.p == 0.6
    assert model.decoder.up_blocks[0][0].res_block.dropout.p == 0.3


@pytest.mark.parametrize(
    "kwargs",
    [{"res_dropout": (0.1, 0.2)}, {"attn_n_heads": (1, 2)}],
    ids=["res", "attn"],
)
def test_throws_error_on_sequence_for_a_shared_argument(kwargs):
    """Test that a shared argument rejects sequences and names the dict to use."""
    with pytest.raises(ValueError, match="encoder_"):
        Autoencoder.build(
            **PER_COMPONENT_CONF,
            encoder_args=PER_COMPONENT_ENC,
            decoder_args=PER_COMPONENT_DEC,
            **kwargs,
        )


INJECT_ENC = {
    "dimensions": 2,
    "in_channels": 1,
    "n_channels": 16,
    "out_channels": 4,
    "down_block_types": ("AutoencoderDownBlock",) * 2,
    "block_out_channel_mults": (1, 2),
    "num_layers_per_block": 1,
    "mid_block_types": (),
    "res_args": {"res_groups": 4},
}
INJECT_DEC = {
    "dimensions": 2,
    "in_channels": 4,
    "n_channels": 32,
    "out_channels": 1,
    "up_block_types": ("AutoencoderUpBlock",) * 2,
    "block_out_channel_mults": (1, 2),
    "num_layers_per_block": 1,
    "mid_block_types": (),
    "res_args": {"res_groups": 4},
}


class _MinimalEncoder(nn.Module):
    """Encoder-like module exposing only what the latent projection reads."""

    def __init__(self, dimensions: int = 2, out_channels: int = 4):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            out_channels: Number of latent channels emitted.
        """
        super().__init__()
        self.dimensions = dimensions
        self.out_channels = out_channels
        self.conv = nn.Conv2d(1, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        """Encode an input tensor.

        Args:
            x: Input tensor.
        """
        return self.conv(x)


class _MinimalDecoder(nn.Module):
    """Decoder-like module exposing only what the latent projection reads."""

    def __init__(self, dimensions: int = 2, in_channels: int = 4):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            in_channels: Number of latent channels consumed.
        """
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.conv = nn.ConvTranspose2d(in_channels, 1, 4, stride=2, padding=1)

    def forward(self, z):
        """Decode a latent tensor.

        Args:
            z: Input latent tensor.
        """
        return self.conv(z)


def test_injected_components_reconstruct_the_input():
    """Test that a model assembled from two components runs end to end."""
    model = Autoencoder(Encoder(**INJECT_ENC), Decoder(**INJECT_DEC))
    sample = torch.randn(1, 1, 16, 16)
    assert model(sample).shape == sample.shape


def test_building_and_injecting_give_the_same_model():
    """Test that both construction paths assemble the same architecture."""
    injected = Autoencoder(Encoder(**INJECT_ENC), Decoder(**INJECT_DEC))
    built = Autoencoder.build(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        latent_dim=4,
        res_groups=4,
        encoder_args={
            k: v
            for k, v in INJECT_ENC.items()
            if k not in ("dimensions", "in_channels", "out_channels")
        },
        decoder_args={
            k: v
            for k, v in INJECT_DEC.items()
            if k not in ("dimensions", "in_channels", "n_channels", "out_channels")
        },
    )
    assert list(injected.state_dict()) == list(built.state_dict())
    assert sum(p.numel() for p in injected.parameters()) == sum(
        p.numel() for p in built.parameters()
    )


def test_components_of_different_rank_are_rejected():
    """Test that the components must agree on the number of dimensions."""
    decoder = Decoder(**{**INJECT_DEC, "dimensions": 3})
    with pytest.raises(ValueError, match="3-dimensional"):
        Autoencoder(Encoder(**INJECT_ENC), decoder)


def test_decoder_that_expects_another_latent_width_is_rejected():
    """Test that a mismatched latent width is caught at construction."""
    decoder = Decoder(**{**INJECT_DEC, "in_channels": 8})
    with pytest.raises(ValueError, match="latent channels"):
        Autoencoder(Encoder(**INJECT_ENC), decoder)


def test_injected_projection_of_the_wrong_width_is_rejected():
    """Test that an injected projection must match the encoder's output."""
    with pytest.raises(ValueError, match="latent_proj"):
        Autoencoder(
            Encoder(**INJECT_ENC),
            Decoder(**INJECT_DEC),
            latent_proj=nn.Conv2d(8, 4, 1),
        )


def test_components_may_implement_only_part_of_the_interface():
    """Test that a module without the full metadata still assembles."""
    model = Autoencoder(_MinimalEncoder(), _MinimalDecoder())
    sample = torch.randn(1, 1, 16, 16)
    assert model(sample).shape == sample.shape
    assert model.latent_dim == 4
    assert model.double_z is False


def test_projections_are_used_as_given_or_omitted():
    """Test that the bottleneck takes a module, a flag, or nothing."""
    proj = nn.Conv2d(4, 4, 1)
    model = Autoencoder(
        Encoder(**INJECT_ENC),
        Decoder(**INJECT_DEC),
        latent_proj=proj,
        latent_deproj=False,
    )
    assert model.latent_proj is proj
    assert model.latent_deproj is None


@pytest.mark.parametrize(
    "component,key,match",
    [
        ("encoder_args", "out_channels", "latent_dim"),
        ("encoder_args", "dimensions", "dimensions"),
        ("decoder_args", "n_channels", "bottleneck"),
        ("decoder_args", "in_channels", "latent width"),
    ],
)
def test_build_rejects_keys_it_sets_itself(component, key, match):
    """Test that a per-component dict cannot override what the model derives."""
    with pytest.raises(ValueError, match=match):
        Autoencoder.build(**{component: {key: 8}})


def test_decoder_width_follows_the_levels_the_encoder_built():
    """Test that surplus channel multipliers do not widen the decoder."""
    model = Autoencoder.build(
        latent_dim=4,
        res_groups=4,
        encoder_args={
            "n_channels": 16,
            "down_block_types": ("AutoencoderDownBlock",) * 2,
            "block_out_channel_mults": (1, 2, 2, 2),
            "num_layers_per_block": 1,
            "mid_block_types": (),
        },
        decoder_args={
            "up_block_types": ("AutoencoderUpBlock",) * 2,
            "block_out_channel_mults": (1, 2),
            "num_layers_per_block": 1,
            "mid_block_types": (),
        },
    )
    assert model.encoder.bottleneck_channels == 32
    assert model.decoder.n_channels == model.encoder.bottleneck_channels

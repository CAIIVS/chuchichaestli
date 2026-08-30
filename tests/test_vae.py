# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Variational autoencoder tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.autoencoder import VAE


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
def test_vae_init(dimensions, in_channels, n_channels, latent_dim, out_channels):
    """Test the VAE module initialization."""
    model = VAE.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels},
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
        (
            2,
            1,
            64,
            4,
            1,
            ("AttnAutoencoderDownBlock",) * 3,
            ("AttnAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("ConvAttnAutoencoderDownBlock",) * 3,
            ("ConvAttnAutoencoderUpBlock",) * 3,
        ),
        (3, 1, 64, 4, 1, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 8, 1, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 4, 3, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vae_blocks(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the VAE module."""
    model = VAE.build(
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
    out, posterior = model(sample)
    assert isinstance(out, torch.Tensor)
    assert isinstance(posterior, torch.distributions.MultivariateNormal)


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
def test_vae_latent_dim(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the VAE module (latent dim)."""
    model = VAE.build(
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
def test_vae_forward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the VAE module (forward pass)."""
    model = VAE.build(
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
    out, posterior = model(sample)
    assert isinstance(out, torch.Tensor)
    assert isinstance(posterior, torch.distributions.MultivariateNormal)
    assert out.shape == (
        1,
        out_channels,
        *(model.f_exp / model.f_comp * wh,) * dimensions,
    )
    print(out.shape, posterior.mode.shape)


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
    """Test the VAE module (backward pass)."""
    model = VAE.build(
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
    out, posterior = model(sample)
    loss = nn.functional.mse_loss(out, sample)
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


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
def test_vae_kl_div(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the VAE kl_divergence."""
    model = VAE.build(
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
    out, posterior = model(sample)
    loss = nn.functional.mse_loss(out, sample)
    kl_div = model.kl_divergence(posterior).sum(dim=1).mean()
    loss += kl_div
    loss.backward()


if __name__ == "__main__":
    pytest.main(["-v", "test_vae.py"])


def test_per_component_args_reach_the_encoder_only():
    """Test that a VAE forwards the per-component override dicts to the right child."""
    model = VAE.build(
        latent_dim=4,
        res_groups=4,
        res_dropout=0.1,
        encoder_args={
            "n_channels": 16,
            "block_out_channel_mults": (1, 2),
            "down_block_types": ("AutoencoderDownBlock",) * 2,
            "mid_block_types": (),
            "num_layers_per_block": 1,
            "res_args": {"res_dropout": (0.3, 0.6)},
        },
        decoder_args={
            "block_out_channel_mults": (1, 2),
            "up_block_types": ("AutoencoderUpBlock",) * 2,
            "mid_block_types": (),
            "num_layers_per_block": 1,
        },
    )
    assert [s[0].res_block.dropout.p for s in model.encoder.down_blocks[::2]] == [
        0.3,
        0.6,
    ]
    assert [s[0].res_block.dropout.p for s in model.decoder.up_blocks[::2]] == [
        0.1,
        0.1,
    ]


def test_vae_requires_an_encoder_that_doubles_the_latent():
    """Test that a VAE refuses an encoder without mean and variance channels."""
    from chuchichaestli.models.autoencoder import Decoder, Encoder

    encoder = Encoder(2, 1, 16, 4, res_args={"res_groups": 4})
    decoder = Decoder(2, 4, encoder.bottleneck_channels, 1, res_args={"res_groups": 4})
    with pytest.raises(ValueError, match="double_z"):
        VAE(encoder, decoder)


def test_vae_encoder_refuses_to_drop_the_doubling():
    """Test that the VAE encoder cannot be built without doubled channels."""
    from chuchichaestli.models.autoencoder import VAEEncoder

    with pytest.raises(ValueError, match="mean and variance"):
        VAEEncoder(2, 1, 16, 4, double_z=False)


def test_vae_builds_its_own_components():
    """Test that the flat constructor picks the VAE-specific components."""
    from chuchichaestli.models.autoencoder import VAEDecoder, VAEEncoder

    model = VAE.build(latent_dim=4, encoder_args={"n_channels": 16})
    assert isinstance(model, VAE)
    assert isinstance(model.encoder, VAEEncoder)
    assert isinstance(model.decoder, VAEDecoder)
    assert model.encoder.out_channels == 8
    assert model.latent_dim == 4


def test_vae_decoder_takes_the_sampled_latent_width():
    """Test that the decoding component consumes the latent code after sampling."""
    from chuchichaestli.models.autoencoder import Decoder, VAEDecoder, VAEEncoder

    encoder = VAEEncoder(2, 1, 16, 4, res_args={"res_groups": 4})
    decoder = VAEDecoder(
        2, 4, encoder.bottleneck_channels, 1, res_args={"res_groups": 4}
    )
    assert isinstance(decoder, Decoder)
    assert decoder.in_channels == encoder.out_channels // 2

    model = VAE(encoder, decoder)
    sample = torch.randn(1, 1, 16, 16)
    out, posterior = model(sample)
    assert out.shape == sample.shape
    assert posterior.mean.shape[1] == decoder.in_channels

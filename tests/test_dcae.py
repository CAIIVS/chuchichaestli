# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standard autoencoder tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.autoencoder import DCAE


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels",
    [
        (2, 1, 64, 4, 1),
        (2, 1, 32, 8, 1),
        (2, 1, 32, 4, 3),
    ],
)
def test_dcae_init(dimensions, in_channels, n_channels, latent_dim, out_channels):
    """Test the VAE module initialization."""
    model = DCAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        attn_groups=8,
        res_groups=8,
        decoder_groups=8,
    )
    assert isinstance(model.encoder, nn.Module)
    assert isinstance(model.decoder, nn.Module)
    assert model.latent_proj is None
    assert model.latent_deproj is None
    assert len(model.encoder.mid_blocks) == 0
    assert len(model.decoder.mid_blocks) == 0


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            128,
            64,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            128,
            32,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            256,
            256,
            1,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            3,
            256,
            64,
            3,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_blocks(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module."""
    model = DCAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        # latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert isinstance(out, torch.Tensor)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            32,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            16,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            1,
            32,
            4,
            3,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_latent_dim(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module (latent dim)."""
    model = DCAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    spatial_dims = (wh // model.f_comp,) * dimensions
    assert model.levels == (len(down_block_types), len(up_block_types))
    assert model.f_comp == 2 ** (len(down_block_types) - 1)
    assert model.f_exp == 2 ** (len(up_block_types) - 1)
    assert model.compute_latent_shape(shape) == (1, latent_dim, *spatial_dims)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            32,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            16,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            1,
            32,
            4,
            3,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_forward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module (forward pass)."""
    model = DCAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (
        1,
        out_channels,
        *(model.f_exp / model.f_comp * wh,) * dimensions,
    )
    print(out.shape)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            32,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            3,
            16,
            8,
            3,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            1,
            32,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_backward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module (backward pass)."""
    model = DCAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    loss = nn.functional.mse_loss(out, sample)
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


def test_dcae_inspect():
    """Test the DCAE module inspection."""
    model = DCAE(
        dimensions=2,
        in_channels=1,
        n_channels=128,
        latent_dim=32,
        out_channels=1,
        # down_block_types=("DCAutoencoderDownBlock",) * 3 + ("EfficientViTBlock",) * 4,
        # up_block_types=("EfficientViTBlock",) * 4 + ("DCAutoencoderUpBlock",) * 3,
        # block_out_channel_mults=(2, 2, 1, 2, 1, 2, 1),
        # attn_scales=(3,5,7),
    )
    try:
        from torchinfo import summary

        summary(
            model,
            (1, 1, 256, 256),
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=8,
        )
    except ImportError:
        print(model)
    print()


if __name__ == "__main__":
    pytest.main(["-v", "test_dcae.py"])

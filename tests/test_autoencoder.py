# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standard autoencoder tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.autoencoder import Autoencoder


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
    model = Autoencoder(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        use_latent_proj=True,
        use_latent_deproj=True,
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
    model = Autoencoder(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        use_latent_proj=True,
        use_latent_deproj=True,
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
    model = Autoencoder(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        use_latent_proj=True,
        use_latent_deproj=True,
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
    model = Autoencoder(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        use_latent_proj=True,
        use_latent_deproj=True,
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
    model = Autoencoder(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        use_latent_proj=True,
        use_latent_deproj=True,
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
    model = Autoencoder(
        dimensions=2,
        in_channels=1,
        n_channels=64,
        latent_dim=4,
        out_channels=1,
        use_latent_proj=True,
        use_latent_deproj=True,
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

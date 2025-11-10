# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Vector-quantized variational autoencoder tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.autoencoder.vqvae import VQVAE, VectorQuantizer


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,vq_dim,down_block_types,up_block_types",
    [
        (1, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vqvae_init(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    vq_dim,
    down_block_types,
    up_block_types,
):
    """Test VQVAE initialization with different parameters."""
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        vq_dim=vq_dim,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    assert isinstance(vqvae.encoder, nn.Module)
    assert isinstance(vqvae.decoder, nn.Module)
    assert isinstance(vqvae.quantize, VectorQuantizer)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,vq_dim,down_block_types,up_block_types",
    [
        (1, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vqvae_encode(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    vq_dim,
    down_block_types,
    up_block_types,
):
    """Test VQVAE initialization with different parameters."""
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        vq_dim=vq_dim,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    embedding_shape = vqvae.compute_embedding_shape(sample.shape)
    z, loss, usage = vqvae.encode(sample)
    assert isinstance(z, torch.Tensor)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(usage, dict)
    assert loss.item() >= 0
    assert z.shape == embedding_shape


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,vq_dim,down_block_types,up_block_types",
    [
        (1, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vqvae_decode(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    vq_dim,
    down_block_types,
    up_block_types,
):
    """Test VQVAE initialization with different parameters."""
    out_channels = in_channels
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        vq_dim=vq_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    embedding_shape = vqvae.compute_embedding_shape(shape)
    sample = torch.randn(embedding_shape)
    out = vqvae.decode(sample)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (
        1,
        out_channels,
        *(vqvae.f_exp / vqvae.f_comp * wh,) * dimensions,
    )


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,vq_dim,down_block_types,up_block_types",
    [
        (1, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vqvae_forward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    vq_dim,
    down_block_types,
    up_block_types,
):
    """Test VQVAE initialization with different parameters."""
    out_channels = in_channels
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        vq_dim=vq_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out, loss, usage = vqvae(sample)
    assert isinstance(out, torch.Tensor)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(usage, dict)
    assert out.shape == (
        1,
        out_channels,
        *(vqvae.f_exp / vqvae.f_comp * wh,) * dimensions,
    )
    assert loss.item() >= 0


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,vq_dim,down_block_types,up_block_types",
    [
        (1, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (2, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (3, 1, 32, 4, 64, ("AutoencoderDownBlock",) * 5, ("AutoencoderUpBlock",) * 5),
        (1, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (2, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (3, 1, 64, 8, 16, ("AutoencoderDownBlock",) * 3, ("AutoencoderUpBlock",) * 3),
        (1, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (2, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (3, 1, 16, 4, 32, ("AutoencoderDownBlock",) * 2, ("AutoencoderUpBlock",) * 2),
        (1, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (2, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
        (3, 1, 32, 8, 32, ("AutoencoderDownBlock",) * 1, ("AutoencoderUpBlock",) * 1),
    ],
)
def test_vqvae_backward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    vq_dim,
    down_block_types,
    up_block_types,
):
    """Test VQVAE initialization with different parameters."""
    out_channels = in_channels
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        vq_dim=vq_dim,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out, vq_loss, _ = vqvae(sample)
    loss = nn.functional.mse_loss(out, sample)
    loss += vq_loss
    loss.backward()
    assert isinstance(vq_loss, torch.Tensor)
    assert loss.item() > 0
    for param in (*vqvae.encoder.parameters(), *vqvae.decoder.parameters()):
        if param.grad is not None:
            assert param.grad.abs().sum() > 0
    for param in vqvae.quantize.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main(["-sv", "test_vqvae.py"])

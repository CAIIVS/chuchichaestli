# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier:  GPL-3.0-or-later
"""Device handling tests for models with potential device mismatch issues."""

import pytest
import torch
from torch import nn


def get_available_devices():
    """Return list of available devices for testing."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("device", get_available_devices())
def test_vae_forward_device(device):
    """Test that VAE forward pass works correctly on different devices."""
    from chuchichaestli.models.autoencoder import VAE

    model = VAE(
        dimensions=2,
        in_channels=1,
        n_channels=32,
        latent_dim=4,
        out_channels=1,
        down_block_types=("AutoencoderDownBlock",) * 2,
        up_block_types=("AutoencoderUpBlock",) * 2,
        use_latent_proj=True,
        use_latent_deproj=True,
    ).to(device)

    sample = torch.randn(1, 1, 16, 16, device=device)
    out, posterior = model(sample)

    assert out.device == device
    assert posterior.mean.device == device


@pytest.mark.parametrize("device", get_available_devices())
def test_vae_kl_divergence_device(device):
    """Test KL divergence computation on different devices (Issue #133)."""
    from chuchichaestli.models.autoencoder import VAE

    model = VAE(
        dimensions=2,
        in_channels=1,
        n_channels=32,
        latent_dim=4,
        out_channels=1,
        down_block_types=("AutoencoderDownBlock",) * 2,
        up_block_types=("AutoencoderUpBlock",) * 2,
    ).to(device)

    sample = torch.randn(1, 1, 16, 16, device=device)
    out, posterior = model(sample)

    # This should not raise a device mismatch error
    kl_div = model.kl_divergence(posterior)

    assert kl_div.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vae_kl_divergence_dtype(device, dtype):
    """Test KL divergence preserves dtype correctly."""
    from chuchichaestli.models.autoencoder import VAE

    model = VAE(
        dimensions=2,
        in_channels=1,
        n_channels=32,
        latent_dim=4,
        out_channels=1,
        down_block_types=("AutoencoderDownBlock",) * 2,
        up_block_types=("AutoencoderUpBlock",) * 2,
    ).to(device=device, dtype=dtype)

    sample = torch.randn(1, 1, 16, 16, device=device, dtype=dtype)
    out, posterior = model(sample)

    kl_div = model.kl_divergence(posterior)

    assert kl_div.dtype == dtype
    assert kl_div.device == device


@pytest.mark.parametrize("device", get_available_devices())
def test_vae_backward_with_kl_device(device):
    """Test backward pass with KL divergence on different devices."""
    from chuchichaestli.models.autoencoder import VAE

    model = VAE(
        dimensions=2,
        in_channels=1,
        n_channels=32,
        latent_dim=4,
        out_channels=1,
        down_block_types=("AutoencoderDownBlock",) * 2,
        up_block_types=("AutoencoderUpBlock",) * 2,
    ).to(device)

    sample = torch.randn(1, 1, 32, 32, device=device)
    out, posterior = model(sample)

    recon_loss = nn.functional.mse_loss(out, sample)
    kl_loss = model.kl_divergence(posterior).sum(dim=1).mean()
    total_loss = recon_loss + kl_loss

    total_loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.device == device


@pytest.mark.parametrize("device", get_available_devices())
def test_gaussian_noise_block_forward_device(device):
    """Test GaussianNoiseBlock forward pass on different devices."""
    from chuchichaestli.models.blocks import GaussianNoiseBlock

    block = GaussianNoiseBlock(sigma=0.1, mu=0.0).to(device)
    block.train()  # Enable training mode for noise generation

    x = torch.randn(2, 3, 8, 8, device=device)
    out = block(x)

    assert out.device == device
    assert out.shape == x.shape

@pytest.mark.parametrize("device", get_available_devices())
def test_gaussian_noise_block_noise_buffer_moves(device):
    """Test that noise buffer moves with module to different devices."""
    from chuchichaestli.models.blocks import GaussianNoiseBlock

    block = GaussianNoiseBlock(sigma=0.1, mu=0.5)

    # Initially on CPU
    assert block.noise.device == torch.device("cpu")

    # Move to target device
    block = block.to(device)

    # Check noise buffer moved
    assert block.noise.device == device

@pytest.mark.parametrize("device", get_available_devices())
def test_gaussian_noise_block_with_nonzero_mu(device):
    """Test GaussianNoiseBlock with non-zero mu on different devices."""
    from chuchichaestli.models.blocks import GaussianNoiseBlock

    block = GaussianNoiseBlock(sigma=0.1, mu=1.0).to(device)
    block.train()

    x = torch.randn(2, 3, 8, 8, device=device)
    out = block(x)

    assert out.device == device

@pytest.mark.parametrize("device", get_available_devices())
def test_gaussian_noise_block_inference_mode(device):
    """Test GaussianNoiseBlock in inference mode (no noise added)."""
    from chuchichaestli.models.blocks import GaussianNoiseBlock

    block = GaussianNoiseBlock(sigma=0.1, mu=0.0).to(device)
    block.eval()

    x = torch.randn(2, 3, 8, 8, device=device)
    out = block(x)

    assert out.device == device
    assert torch.equal(out, x)  # No noise should be added in eval mode



@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("add_noise", ["up", "down"])
def test_unet_with_noise_device(device, add_noise):
    """Test UNet with noise blocks on different devices."""
    from chuchichaestli.models.unet import UNet

    model = UNet(
        dimensions=2,
        in_channels=1,
        n_channels=16,
        out_channels=1,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2),
        add_noise=add_noise,
        noise_sigma=0.1,
        res_groups=8,
    ).to(device)

    model.train()
    x = torch.randn(1, 1, 16, 16, device=device)
    out = model(x)

    assert out.device == device
    assert out.shape == x.shape


@pytest.mark.parametrize("device", get_available_devices())
def test_sinusoidal_embedding_device(device):
    """Test SinusoidalTimeEmbedding on different devices."""
    from chuchichaestli.models.unet.time_embeddings import SinusoidalTimeEmbedding

    emb = SinusoidalTimeEmbedding(num_channels=32)
    timesteps = torch.tensor([0.1, 0.5, 0.9], device=device)

    out = emb(timesteps)

    assert out.device == device
    assert out.shape == (3, 32)

@pytest.mark.parametrize("device", get_available_devices())
def test_deep_sinusoidal_embedding_device(device):
    """Test DeepSinusoidalTimeEmbedding on different devices."""
    from chuchichaestli.models.unet.time_embeddings import (
        DeepSinusoidalTimeEmbedding,
    )

    emb = DeepSinusoidalTimeEmbedding(num_channels=32).to(device)
    timesteps = torch.tensor([0.1, 0.5, 0.9], device=device)

    out = emb(timesteps)

    assert out.device == device
    assert out.shape == (3, 32)

@pytest.mark.parametrize("device", get_available_devices())
def test_unet_with_time_embedding_device(device):
    """Test UNet with time embeddings on different devices."""
    from chuchichaestli.models.unet import UNet

    model = UNet(
        dimensions=2,
        in_channels=1,
        n_channels=16,
        out_channels=1,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2),
        time_embedding="SinusoidalTimeEmbedding",
        time_channels=16,
        res_groups=8,
    ).to(device)

    x = torch.randn(1, 1, 16, 16, device=device)
    t = torch.tensor([0.5], device=device)

    out = model(x, t)

    assert out.device == device

@pytest.mark.parametrize("device", get_available_devices())
def test_unet_with_scalar_timestep_device(device):
    """Test UNet with scalar timestep on different devices."""
    from chuchichaestli.models.unet import UNet

    model = UNet(
        dimensions=2,
        in_channels=1,
        n_channels=16,
        out_channels=1,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2),
        time_embedding="SinusoidalTimeEmbedding",
        time_channels=16,
        res_groups=8,
    ).to(device)

    x = torch.randn(1, 1, 16, 16, device=device)

    # Scalar timestep (not tensor) should be converted to tensor on correct device
    out = model(x, 0.5)

    assert out.device == device


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("device", get_available_devices())
def test_vae_dimensions_device(dimensions, device):
    """Test VAE with different dimensions on different devices."""
    from chuchichaestli.models.autoencoder import VAE

    model = VAE(
        dimensions=dimensions,
        in_channels=1,
        n_channels=16,
        latent_dim=4,
        out_channels=1,
        down_block_types=("AutoencoderDownBlock",) * 2,
        up_block_types=("AutoencoderUpBlock",) * 2,
    ).to(device)

    wh = 16
    shape = (1, 1) + (wh,) * dimensions
    sample = torch.randn(shape, device=device)

    out, posterior = model(sample)
    kl_div = model.kl_divergence(posterior)

    assert out.device == device
    assert posterior.mean.device == device
    assert kl_div.device == device

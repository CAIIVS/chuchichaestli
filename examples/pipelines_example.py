#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example demonstrating the use of the pipelines module.

This example shows how to use the pipelines module to create
composite pipeline classes through multiple inheritance.
"""

import torch
from torch import nn
from torch.optim import Adam

from chuchichaestli.pipelines import (
    AutoencoderMixin,
    DiffusionMixin,
    Trainer,
    AutoencoderInferencePipeline,
)


# Simple encoder/decoder models for demonstration
class SimpleEncoder(nn.Module):
    """Simple encoder model."""

    def __init__(self, in_dim=784, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.fc(x)


class SimpleDecoder(nn.Module):
    """Simple decoder model."""

    def __init__(self, latent_dim=32, out_dim=784):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.fc(z)


def example_inference_pipeline():
    """Example of using AutoencoderInferencePipeline."""
    print("=" * 60)
    print("Example 1: AutoencoderInferencePipeline")
    print("=" * 60)

    # Create encoder and decoder
    encoder = SimpleEncoder(in_dim=784, latent_dim=32)
    decoder = SimpleDecoder(latent_dim=32, out_dim=784)

    # Create inference pipeline
    pipeline = AutoencoderInferencePipeline(
        encoder=encoder, decoder=decoder, device="cpu"
    )

    # Generate sample data
    x = torch.randn(4, 784)

    # Encode
    z = pipeline.predict(x, mode="encode")
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")

    # Decode
    x_decoded = pipeline.predict(z, mode="decode")
    print(f"Decoded shape: {x_decoded.shape}")

    # Reconstruct
    x_recon = pipeline(x, mode="reconstruct")
    print(f"Reconstructed shape: {x_recon.shape}")
    print()


def example_composite_trainer():
    """Example of creating a composite trainer with multiple mixins."""
    print("=" * 60)
    print("Example 2: Composite Trainer with Multiple Mixins")
    print("=" * 60)

    class LatentDiffusionTrainer(AutoencoderMixin, DiffusionMixin, Trainer):
        """A trainer combining autoencoder, diffusion, and base training functionality.

        This demonstrates the composability of the pipeline module through
        multiple inheritance.
        """

        def __init__(self, encoder, decoder, diffusion_model, optimizer):
            super().__init__(
                encoder=encoder,
                decoder=decoder,
                diffusion_model=diffusion_model,
                noise_scheduler=None,
                model=diffusion_model,
                optimizer=optimizer,
                device="cpu",
            )

        def train_step(self, batch, *args, **kwargs):
            """Minimal training step for demonstration."""
            # In a real implementation, this would include:
            # 1. Encode input to latent space
            # 2. Add noise to latents
            # 3. Predict noise with diffusion model
            # 4. Compute loss and update
            return {"loss": 0.0}

    # Create components
    encoder = SimpleEncoder(in_dim=784, latent_dim=32)
    decoder = SimpleDecoder(latent_dim=32, out_dim=784)
    diffusion_model = nn.Linear(32, 32)  # Simplified for demonstration
    optimizer = Adam(diffusion_model.parameters(), lr=0.001)

    # Create trainer
    trainer = LatentDiffusionTrainer(
        encoder=encoder,
        decoder=decoder,
        diffusion_model=diffusion_model,
        optimizer=optimizer,
    )

    print(f"Trainer class: {trainer.__class__.__name__}")
    print(f"Base classes: {[c.__name__ for c in trainer.__class__.__bases__]}")
    print(f"Has encoder: {trainer.encoder is not None}")
    print(f"Has decoder: {trainer.decoder is not None}")
    print(f"Has diffusion_model: {trainer.diffusion_model is not None}")

    # Demonstrate encoding/decoding
    x = torch.randn(4, 784)
    z = trainer.encode(x)
    x_recon = trainer.decode(z)
    print(f"\nInput shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print()


if __name__ == "__main__":
    print("\nchuchichaestli.pipelines Module Examples")
    print("=" * 60)
    print()

    example_inference_pipeline()
    example_composite_trainer()

    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)

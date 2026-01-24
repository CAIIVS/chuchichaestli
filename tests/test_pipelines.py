# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the pipelines module."""

import pytest
import torch
from torch import nn

from chuchichaestli.pipelines import (
    InferencePipeline,
    Trainer,
    AutoencoderMixin,
    AutoencoderInferencePipeline,
    EncoderMixin,
    EncoderInferencePipeline,
    DecoderMixin,
    DecoderInferencePipeline,
    DiffusionMixin,
    DiffusionInferencePipeline,
    CGANMixin,
    CGANInferencePipeline,
    CGANTrainer,
)


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


def test_imports():
    """Test that all required classes can be imported."""
    assert InferencePipeline is not None
    assert Trainer is not None
    assert AutoencoderMixin is not None
    assert AutoencoderInferencePipeline is not None
    assert EncoderMixin is not None
    assert EncoderInferencePipeline is not None
    assert DecoderMixin is not None
    assert DecoderInferencePipeline is not None
    assert DiffusionMixin is not None
    assert DiffusionInferencePipeline is not None
    assert CGANMixin is not None
    assert CGANInferencePipeline is not None
    assert CGANTrainer is not None


def test_autoencoder_mixin():
    """Test AutoencoderMixin functionality."""

    class TestAutoencoder(AutoencoderMixin):
        def __init__(self):
            encoder = SimpleModel(10, 5)
            decoder = SimpleModel(5, 10)
            super().__init__(encoder=encoder, decoder=decoder)

    ae = TestAutoencoder()
    assert ae.encoder is not None
    assert ae.decoder is not None

    # Test encode
    x = torch.randn(2, 10)
    z = ae.encode(x)
    assert z.shape == (2, 5)

    # Test decode
    x_recon = ae.decode(z)
    assert x_recon.shape == (2, 10)

    # Test reconstruct
    x_recon2 = ae.reconstruct(x)
    assert x_recon2.shape == (2, 10)


def test_encoder_mixin():
    """Test EncoderMixin functionality."""

    class TestEncoder(EncoderMixin):
        def __init__(self):
            encoder = SimpleModel(10, 5)
            super().__init__(encoder=encoder)

    enc = TestEncoder()
    assert enc.encoder is not None

    # Test encode
    x = torch.randn(2, 10)
    z = enc.encode(x)
    assert z.shape == (2, 5)


def test_decoder_mixin():
    """Test DecoderMixin functionality."""

    class TestDecoder(DecoderMixin):
        def __init__(self):
            decoder = SimpleModel(5, 10)
            super().__init__(decoder=decoder)

    dec = TestDecoder()
    assert dec.decoder is not None

    # Test decode
    z = torch.randn(2, 5)
    x = dec.decode(z)
    assert x.shape == (2, 10)


def test_autoencoder_inference_pipeline():
    """Test AutoencoderInferencePipeline."""
    encoder = SimpleModel(10, 5)
    decoder = SimpleModel(5, 10)

    pipeline = AutoencoderInferencePipeline(
        encoder=encoder, decoder=decoder, device="cpu"
    )

    x = torch.randn(2, 10)

    # Test encode mode
    z = pipeline.predict(x, mode="encode")
    assert z.shape == (2, 5)

    # Test decode mode
    x_decoded = pipeline.predict(z, mode="decode")
    assert x_decoded.shape == (2, 10)

    # Test reconstruct mode
    x_recon = pipeline.predict(x, mode="reconstruct")
    assert x_recon.shape == (2, 10)

    # Test __call__
    x_recon2 = pipeline(x, mode="reconstruct")
    assert x_recon2.shape == (2, 10)


def test_encoder_inference_pipeline():
    """Test EncoderInferencePipeline."""
    encoder = SimpleModel(10, 5)

    pipeline = EncoderInferencePipeline(encoder=encoder, device="cpu")

    x = torch.randn(2, 10)
    z = pipeline.predict(x)
    assert z.shape == (2, 5)

    # Test __call__
    z2 = pipeline(x)
    assert z2.shape == (2, 5)


def test_decoder_inference_pipeline():
    """Test DecoderInferencePipeline."""
    decoder = SimpleModel(5, 10)

    pipeline = DecoderInferencePipeline(decoder=decoder, device="cpu")

    z = torch.randn(2, 5)
    x = pipeline.predict(z)
    assert x.shape == (2, 10)

    # Test __call__
    x2 = pipeline(z)
    assert x2.shape == (2, 10)


def test_cgan_mixin():
    """Test CGANMixin functionality."""

    class TestCGAN(CGANMixin):
        def __init__(self):
            generator = SimpleModel(10, 5)
            discriminator = SimpleModel(5, 1)
            super().__init__(generator=generator, discriminator=discriminator)

    cgan = TestCGAN()
    assert cgan.generator is not None
    assert cgan.discriminator is not None

    # Test generate
    noise = torch.randn(2, 10)
    samples = cgan.generate(noise)
    assert samples.shape == (2, 5)

    # Test discriminate
    x = torch.randn(2, 5)
    pred = cgan.discriminate(x)
    assert pred.shape == (2, 1)


def test_multiple_inheritance():
    """Test that mixins can be combined through multiple inheritance."""

    class LatentDiffusionTrainer(AutoencoderMixin, DiffusionMixin, Trainer):
        """Combined trainer using multiple mixins."""

        def __init__(self):
            encoder = SimpleModel(10, 5)
            decoder = SimpleModel(5, 10)
            model = SimpleModel(10, 10)
            super().__init__(
                encoder=encoder,
                decoder=decoder,
                diffusion_model=None,
                noise_scheduler=None,
                model=model,
                optimizer=None,
                device="cpu",
            )

        def train_step(self, batch, *args, **kwargs):
            # Minimal implementation for testing
            return {"loss": 0.0}

    trainer = LatentDiffusionTrainer()

    # Test that all components are accessible
    assert trainer.encoder is not None
    assert trainer.decoder is not None
    assert trainer.model is not None

    # Test encoder/decoder functionality
    x = torch.randn(2, 10)
    z = trainer.encode(x)
    assert z.shape == (2, 5)

    x_recon = trainer.decode(z)
    assert x_recon.shape == (2, 10)


def test_cgan_inference_pipeline():
    """Test CGANInferencePipeline."""
    generator = SimpleModel(100, 5)

    pipeline = CGANInferencePipeline(generator=generator, device="cpu")

    # Test prediction
    samples = pipeline.predict(batch_size=2, latent_dim=100)
    assert samples.shape == (2, 5)

    # Test __call__
    samples2 = pipeline(batch_size=3, latent_dim=100)
    assert samples2.shape == (3, 5)

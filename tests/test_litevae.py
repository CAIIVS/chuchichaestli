# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the wavelet-domain variational autoencoder."""

import pytest
import torch
from torch.distributions import MultivariateNormal

from chuchichaestli.models.autoencoder import (
    DecoderLike,
    EncoderLike,
    LiteVAE,
    LiteVAEDecoder,
    LiteVAEEncoder,
    LiteVAE_B,
    LiteVAE_L,
    LiteVAE_M,
    LiteVAE_S,
)
from chuchichaestli.models.smc import SMConvND


DIMENSIONS = [1, 2, 3]

# The published variants, with the encoder sizes they report in millions of
# parameters.
VARIANTS = {
    LiteVAE_S: 1.03,
    LiteVAE_B: 6.75,
    LiteVAE_M: 32.75,
    LiteVAE_L: 41.42,
}


def tiny(dimensions: int, **kwargs) -> LiteVAE:
    """Build a small model of the given rank, cheap enough to run in a test."""
    return LiteVAE.build(
        dimensions=dimensions,
        in_channels=2,
        out_channels=2,
        latent_dim=4,
        decoder_n_channels=32,
        encoder_args={"n_channels": 8, "aggregator_channels": 8},
        decoder_args={"block_out_channel_mults": (1, 1, 2, 2)},
        **kwargs,
    )


class TestEncoder:
    """Tests for the wavelet-domain encoding component."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_satisfies_the_encoder_protocol(self, dimensions):
        """Test that the encoder can stand in for one without inheriting from it."""
        encoder = LiteVAEEncoder(dimensions=dimensions, in_channels=2, out_channels=4)
        assert isinstance(encoder, EncoderLike)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    @pytest.mark.parametrize("dwt_levels", [1, 2, 3])
    def test_the_wavelet_levels_set_the_compression(self, dimensions, dwt_levels):
        """Test that all spatial compression comes from the wavelet transform."""
        encoder = LiteVAEEncoder(
            dimensions=dimensions, in_channels=2, out_channels=4, dwt_levels=dwt_levels
        )
        assert encoder.f == 2**dwt_levels
        assert encoder.levels == dwt_levels
        size = 8 * 2**dwt_levels
        out = encoder(torch.randn(1, 2, *([size] * dimensions)))
        assert out.shape == (1, 8, *([size // 2**dwt_levels] * dimensions))

    @pytest.mark.parametrize("mults", [(2, 1), (3, 1), (2, 2), (2, 3, 2)])
    def test_the_bottleneck_width_is_the_one_the_aggregator_reaches(self, mults):
        """Test that the reported bottleneck matches the aggregator's widest block."""
        encoder = LiteVAEEncoder(
            dimensions=2,
            in_channels=2,
            out_channels=4,
            aggregator_channels=8,
            aggregator_channel_mults=mults,
        )
        mid = encoder.aggregator.mid_block.res_block.conv1
        assert encoder.bottleneck_channels == mid.weight.shape[1]

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_doubles_its_latent_channels(self, dimensions):
        """Test the mean and variance packing a variational encoder owes."""
        encoder = LiteVAEEncoder(dimensions=dimensions, in_channels=2, out_channels=5)
        assert encoder.latent_channels == 5
        assert encoder.out_channels == 10

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_one_extractor_per_wavelet_level(self, dimensions):
        """Test that every level is processed by its own network."""
        encoder = LiteVAEEncoder(
            dimensions=dimensions, in_channels=2, out_channels=4, dwt_levels=3
        )
        assert len(encoder.extractors) == 3
        assert len(encoder.pools) == 3

    def test_the_transform_is_detached(self):
        """Test that the decomposition carries no gradient of its own."""
        encoder = LiteVAEEncoder(dimensions=2, in_channels=2, out_channels=4)
        bands = encoder.dwt(torch.randn(1, 2, 32, 32, requires_grad=True))
        assert all(band.grad_fn is not None for band in bands)
        with torch.no_grad():
            assert all(b.grad_fn is None for b in encoder.dwt(torch.randn(1, 2, 32, 32)))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_reach_every_extractor(self, dimensions):
        """Test that no level is left out of the backward pass."""
        encoder = LiteVAEEncoder(dimensions=dimensions, in_channels=2, out_channels=4)
        encoder(torch.randn(1, 2, *([32] * dimensions))).pow(2).sum().backward()
        for level, extractor in enumerate(encoder.extractors):
            grads = [p.grad for p in extractor.parameters() if p.grad is not None]
            assert grads, f"extractor {level} received no gradient"

    def test_a_non_positive_level_count_raises(self):
        """Test that the encoder needs at least one wavelet level."""
        with pytest.raises(ValueError, match="at least one level"):
            LiteVAEEncoder(dimensions=2, dwt_levels=0)


class TestDecoder:
    """Tests for the decoding component."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_satisfies_the_decoder_protocol(self, dimensions):
        """Test that the decoder can stand in for one."""
        decoder = LiteVAEDecoder(
            dimensions=dimensions, in_channels=4, n_channels=32, out_channels=2
        )
        assert isinstance(decoder, DecoderLike)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_normalizations_are_self_modulated_convolutions(self, dimensions):
        """Test that the residual blocks carry no normalization of their own."""
        decoder = LiteVAEDecoder(
            dimensions=dimensions, in_channels=4, n_channels=32, out_channels=2
        )
        assert any(isinstance(m, SMConvND) for m in decoder.modules())
        assert isinstance(decoder.out_block.conv, SMConvND)
        norms = [
            type(m).__name__
            for m in decoder.up_blocks.modules()
            if "Norm" in type(m).__name__
        ]
        assert norms == []

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_expands_by_its_levels(self, dimensions):
        """Test the spatial expansion of the decoder."""
        decoder = LiteVAEDecoder(
            dimensions=dimensions, in_channels=4, n_channels=32, out_channels=2
        )
        x = torch.randn(1, 4, *([4] * dimensions))
        assert decoder(x).shape == (1, 2, *([4 * decoder.f] * dimensions))


class TestLiteVAE:
    """Tests for the assembled model."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_a_forward_pass_round_trips(self, dimensions):
        """Test that the model reconstructs the shape it was given."""
        model = tiny(dimensions)
        x = torch.randn(1, 2, *([32] * dimensions))
        recon, posterior = model(x)
        assert recon.shape == x.shape
        assert isinstance(posterior, MultivariateNormal)
        assert posterior.mean.shape == (1, 4, *([4] * dimensions))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_latent_shape_is_predicted(self, dimensions):
        """Test that the model can state its latent shape without running."""
        model = tiny(dimensions)
        shape = (1, 2, *([32] * dimensions))
        assert model.compute_latent_shape(shape) == (1, 4, *([4] * dimensions))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_divergence_is_finite(self, dimensions):
        """Test the regularization term of the loss."""
        model = tiny(dimensions)
        _, posterior = model(torch.randn(1, 2, *([32] * dimensions)))
        assert torch.isfinite(model.kl_divergence(posterior)).all()

    @pytest.mark.parametrize("sample_posterior", [True, False])
    def test_both_latent_paths_run(self, sample_posterior):
        """Test sampling the posterior and taking its mode."""
        model = tiny(2)
        recon, _ = model(torch.randn(1, 2, 32, 32), sample_posterior=sample_posterior)
        assert recon.shape == (1, 2, 32, 32)

    def test_the_latent_projections_are_omitted_by_default(self):
        """Test that the pointwise projections are left out, as the design wants."""
        model = tiny(2)
        assert model.latent_proj is None
        assert model.latent_deproj is None

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_reach_every_parameter(self, dimensions):
        """Test that the whole model trains."""
        model = tiny(dimensions)
        x = torch.randn(1, 2, *([32] * dimensions))
        recon, posterior = model(x)
        (torch.nn.functional.mse_loss(recon, x) + model.kl_divergence(posterior).mean()).backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing == []


class TestVariants:
    """Tests for the subclasses carrying the published encoder sizes."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_variant_builds(self, variant):
        """Test that each published size produces a working model."""
        model = variant.build(
            dimensions=2, in_channels=3, out_channels=3, latent_dim=12
        )
        assert model(torch.randn(1, 3, 32, 32))[0].shape == (1, 3, 32, 32)

    @pytest.mark.parametrize("variant,published", VARIANTS.items())
    def test_the_encoder_sizes_track_the_published_ones(self, variant, published):
        """Test the encoder parameter count against the published variant.

        The multipliers are successive ratios, so a published ladder of widths
        is reproduced exactly where those ratios are whole numbers. `B` needs a
        ratio of 1.5, which no integer reaches, so it is approximated; a fifth
        either way covers both cases.
        """
        model = variant.build(
            dimensions=2, in_channels=3, out_channels=3, latent_dim=12
        )
        millions = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        assert millions == pytest.approx(published, rel=0.2)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_variant_is_a_litevae(self, variant):
        """Test that the variants only bind an encoder size."""
        assert issubclass(variant, LiteVAE)
        assert variant.decoder_cls is LiteVAE.decoder_cls

    def test_the_large_variant_only_widens_the_aggregator(self):
        """Test that `L` differs from `M` in its aggregator alone."""
        medium = LiteVAE_M.build(dimensions=2, in_channels=3, out_channels=3, latent_dim=12)
        large = LiteVAE_L.build(dimensions=2, in_channels=3, out_channels=3, latent_dim=12)
        extractors = [
            sum(p.numel() for p in m.encoder.extractors.parameters())
            for m in (medium, large)
        ]
        aggregators = [
            sum(p.numel() for p in m.encoder.aggregator.parameters())
            for m in (medium, large)
        ]
        assert extractors[0] == extractors[1]
        assert aggregators[0] < aggregators[1]

    def test_the_variants_are_ordered_by_size(self):
        """Test that the variants grow with their name."""
        sizes = [
            sum(
                p.numel()
                for p in variant.build(
                    dimensions=2, in_channels=3, out_channels=3, latent_dim=12
                ).encoder.parameters()
            )
            for variant in VARIANTS
        ]
        assert sizes == sorted(sizes)

    def test_a_variant_preset_can_be_overridden(self):
        """Test that a bound size is a starting point rather than a lock."""
        model = LiteVAE_S.build(
            dimensions=2,
            in_channels=3,
            out_channels=3,
            latent_dim=12,
            encoder_args={"n_channels": 8},
        )
        assert model.encoder.n_channels == 8

    def test_the_base_class_takes_no_preset(self):
        """Test that `LiteVAE` itself is configured entirely by its arguments."""
        model = LiteVAE.build(
            dimensions=2,
            in_channels=3,
            out_channels=3,
            latent_dim=12,
            encoder_args={"n_channels": 24},
        )
        assert model.encoder.n_channels == 24

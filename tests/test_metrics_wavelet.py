# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the high-frequency reconstruction losses."""

import pytest
import torch

from chuchichaestli.metrics import GaussianLoss, WaveletLoss
from chuchichaestli.metrics.functional import charbonnier


DIMENSIONS = [1, 2, 3]


def noisy(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Return `x` with white noise of the given scale added."""
    return x + scale * torch.randn_like(x)


class TestCharbonnier:
    """Tests for the smooth stand-in for the absolute error."""

    def test_it_floors_at_eps_where_the_inputs_agree(self):
        """Test that the penalty is smooth rather than zero at its minimum."""
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        assert float(charbonnier(x, x, eps=1e-3)) == pytest.approx(1e-3)

    def test_it_approaches_the_absolute_error(self):
        """Test that a small `eps` leaves the absolute error behind."""
        x = torch.zeros(4, dtype=torch.float64)
        y = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64)
        assert float(charbonnier(x, y, eps=1e-9)) == pytest.approx(2.5, abs=1e-6)

    def test_it_is_differentiable_at_zero(self):
        """Test the property `eps` exists for, which the absolute error lacks."""
        x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        charbonnier(x, torch.zeros(1, dtype=torch.float64)).backward()
        assert torch.isfinite(x.grad).all()
        assert float(x.grad) == pytest.approx(0.0)

    def test_the_reduction_can_be_dropped(self):
        """Test that the unreduced penalty keeps the input shape."""
        x, y = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
        assert charbonnier(x, y, reduction=None).shape == x.shape
        assert charbonnier(x, y, reduction=torch.sum).ndim == 0


class TestWaveletLoss:
    """Tests for the penalty on the detail subbands."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_is_minimal_where_the_inputs_agree(self, dimensions):
        """Test that identical inputs sit at the floor of the penalty."""
        loss = WaveletLoss(dimensions, levels=2, eps=1e-3)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        assert float(loss(x, x)) == pytest.approx(1e-3)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_grows_with_the_error(self, dimensions):
        """Test that the loss is monotone in the noise it is given."""
        torch.manual_seed(0)
        loss = WaveletLoss(dimensions, levels=1)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        values = [float(loss(x, noisy(x, scale))) for scale in (0.05, 0.1, 0.2, 0.4)]
        assert values == sorted(values)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_ignores_a_constant_offset(self, dimensions):
        """Test that only the detail bands are penalized.

        A constant offset lands entirely in the approximation band, so a loss
        that reads only the detail bands has to stay at its floor.
        """
        loss = WaveletLoss(dimensions, levels=1, eps=1e-3)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        assert float(loss(x, x + 5.0)) == pytest.approx(1e-3)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_registers_high_frequency_error(self, dimensions):
        """Test that a perturbation the detail bands carry is penalized."""
        loss = WaveletLoss(dimensions, levels=1, eps=1e-3)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        alternating = x.clone()
        alternating[(...,) + (slice(None, None, 2),) * dimensions] += 0.5
        assert float(loss(x, alternating)) > 0.1

    @pytest.mark.parametrize("levels", [1, 2, 3])
    def test_every_level_is_penalized(self, levels):
        """Test that the loss reads as many levels as it was built for."""
        loss = WaveletLoss(2, levels=levels)
        assert loss.levels == levels
        x = torch.randn(1, 2, 32, 32, dtype=torch.float64)
        unreduced = loss(x, noisy(x, 0.1), reduction=None)
        expected = sum(2 * 3 * (32 // 2**level) ** 2 for level in range(1, levels + 1))
        assert unreduced.shape == (1, expected)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_reach_the_prediction(self, dimensions):
        """Test that the loss trains the reconstruction it is given."""
        loss = WaveletLoss(dimensions, levels=2)
        x = torch.randn(2, 3, *([16] * dimensions))
        prediction = noisy(x, 0.1).requires_grad_(True)
        loss(x, prediction).backward()
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert float(prediction.grad.abs().sum()) > 0

    def test_the_reduction_can_be_overridden_per_call(self):
        """Test that the reduction follows `SSIMLoss`."""
        loss = WaveletLoss(2, levels=1)
        x = torch.randn(1, 2, 16, 16)
        assert loss(x, x, reduction=torch.sum).ndim == 0
        assert loss(x, x, reduction=None).ndim == 2

    def test_it_holds_no_trainable_state(self):
        """Test that the wavelet filters are constants."""
        assert list(WaveletLoss(2).parameters()) == []

    def test_mismatched_shapes_raise(self):
        """Test that the two inputs have to line up."""
        loss = WaveletLoss(2)
        with pytest.raises(ValueError, match="same shape"):
            loss(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 8, 8))


class TestGaussianLoss:
    """Tests for the penalty on the blur residual."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_vanishes_where_the_inputs_agree(self, dimensions):
        """Test that identical inputs cost nothing."""
        loss = GaussianLoss(dimensions)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        assert float(loss(x, x)) == pytest.approx(0.0)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_grows_with_the_error(self, dimensions):
        """Test that the loss is monotone in the noise it is given."""
        torch.manual_seed(0)
        loss = GaussianLoss(dimensions)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        values = [float(loss(x, noisy(x, scale))) for scale in (0.05, 0.1, 0.2, 0.4)]
        assert values == sorted(values)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_ignores_a_constant_offset(self, dimensions):
        """Test that subtracting the blur removes what both images share."""
        loss = GaussianLoss(dimensions)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        assert float(loss(x, x + 5.0)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_it_registers_high_frequency_error(self, dimensions):
        """Test that a perturbation the blur removes is penalized."""
        loss = GaussianLoss(dimensions)
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        alternating = x.clone()
        alternating[(...,) + (slice(None, None, 2),) * dimensions] += 0.5
        assert float(loss(x, alternating)) > 0.05

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_reach_the_prediction(self, dimensions):
        """Test that the loss trains the reconstruction it is given."""
        loss = GaussianLoss(dimensions)
        x = torch.randn(2, 3, *([16] * dimensions))
        prediction = noisy(x, 0.1).requires_grad_(True)
        loss(x, prediction).backward()
        assert prediction.grad is not None
        assert float(prediction.grad.abs().sum()) > 0

    def test_the_reduction_can_be_overridden_per_call(self):
        """Test that the reduction follows `SSIMLoss`."""
        loss = GaussianLoss(2)
        x = torch.randn(1, 2, 16, 16)
        assert loss(x, x, reduction=torch.sum).ndim == 0
        assert loss(x, x, reduction=None).shape == x.shape

    def test_it_holds_no_trainable_state(self):
        """Test that the Gaussian kernel is a constant."""
        assert list(GaussianLoss(2).parameters()) == []

    def test_mismatched_shapes_raise(self):
        """Test that the two inputs have to line up."""
        loss = GaussianLoss(2)
        with pytest.raises(ValueError, match="same shape"):
            loss(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 8, 8))


class TestTogether:
    """Tests for the two losses used as a training objective would use them."""

    def test_they_combine_into_one_objective(self):
        """Test that both losses backpropagate through one reconstruction."""
        from chuchichaestli.models.autoencoder import LiteVAE_S

        model = LiteVAE_S.build(
            dimensions=2, in_channels=2, out_channels=2, latent_dim=4,
            decoder_n_channels=32,
            decoder_args={"block_out_channel_mults": (1, 1, 2, 2)},
        )
        x = torch.randn(1, 2, 32, 32)
        recon, posterior = model(x)
        objective = (
            torch.nn.functional.mse_loss(recon, x)
            + WaveletLoss(2, levels=2)(x, recon)
            + GaussianLoss(2)(x, recon)
            + 1e-6 * model.kl_divergence(posterior).mean()
        )
        objective.backward()
        assert all(p.grad is not None for p in model.parameters())

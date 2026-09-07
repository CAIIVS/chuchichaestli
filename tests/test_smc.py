# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the self-modulated convolutions and the Gaussian blur."""

from typing import get_args

import pytest
import torch

from chuchichaestli.models.blocks import (
    BLOCK_MAP,
    CONV_BLOCK_MAP,
    RESIDUAL_BLOCK_MAP,
    ConvBlockTypes,
    ResidualBlockTypes,
    SMConvBlock,
    SMConvResidualBlock,
)
from chuchichaestli.models.blur import GaussianBlurND, gaussian_kernel_1d
from chuchichaestli.models.smc import SMConvND


DIMENSIONS = [1, 2, 3]


class TestSMConvND:
    """Tests for the self-modulated convolution."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_shapes(self, dimensions):
        """Test that the convolution maps between the requested channel counts."""
        layer = SMConvND(dimensions, 6, 8)
        out = layer(torch.randn(2, 6, *([12] * dimensions)))
        assert out.shape == (2, 8, *([12] * dimensions))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_weight_is_demodulated_to_unit_norm(self, dimensions):
        """Test the property the demodulation exists to guarantee."""
        layer = SMConvND(dimensions, 5, 7)
        with torch.no_grad():
            layer.scales.normal_()
        weight = layer.modulated_weight() / layer.gain
        axes = tuple(range(1, dimensions + 2))
        norms = weight.pow(2).sum(dim=axes).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_scales_change_the_weight(self, dimensions):
        """Test that the modulation actually reaches the weight."""
        layer = SMConvND(dimensions, 4, 4)
        before = layer.modulated_weight().clone()
        with torch.no_grad():
            layer.scales[0] *= 5.0
        assert not torch.allclose(before, layer.modulated_weight())

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_reach_every_parameter(self, dimensions):
        """Test that the weight, bias, scales and gain all train."""
        layer = SMConvND(dimensions, 4, 4)
        layer(torch.randn(2, 4, *([8] * dimensions))).pow(2).sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_a_stride_downsamples(self, dimensions):
        """Test that the convolution honours its stride."""
        layer = SMConvND(dimensions, 4, 4, stride=2, padding=1)
        out = layer(torch.randn(1, 4, *([8] * dimensions)))
        assert out.shape == (1, 4, *([4] * dimensions))

    @pytest.mark.parametrize("bias", [True, False])
    def test_the_bias_is_optional(self, bias):
        """Test that the convolution runs with and without a bias."""
        layer = SMConvND(2, 4, 4, bias=bias)
        assert (layer.bias is not None) is bias
        assert layer(torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8)

    def test_the_output_channels_default_to_the_input(self):
        """Test the default channel count."""
        assert SMConvND(2, 6).out_channels == 6

    def test_repr_names_the_shape(self):
        """Test that the representation carries the identifying fields."""
        assert "6, 8" in repr(SMConvND(2, 6, 8))


    def test_a_dilated_convolution_is_actually_dilated(self):
        """Test that a convolution keyword survives into the forward pass."""
        import torch.nn.functional as F

        conv = SMConvND(2, 4, 4, kernel_size=3, padding=1, dilation=2)
        x = torch.randn(1, 4, 16, 16)
        want = F.conv2d(x, conv.modulated_weight(), stride=1, padding=1, dilation=2)
        if conv.bias is not None:
            want = want + conv.bias.view(1, -1, 1, 1)
        assert torch.allclose(conv(x, torch.zeros(1, 4)), want, atol=1e-6)

    @pytest.mark.parametrize("kwargs", [{"groups": 2}, {"padding_mode": "reflect"}])
    def test_a_keyword_the_forward_pass_drops_raises(self, kwargs):
        """Test that a silently ignored convolution keyword is refused."""
        with pytest.raises(ValueError, match="cannot honour"):
            SMConvND(2, 4, 4, kernel_size=3, **kwargs)

    @pytest.mark.parametrize("collapse", ["scales", "all"])
    def test_collapsed_weights_do_not_make_nan(self, collapse):
        """Test the demodulation where an inverse square root would divide by zero."""
        conv = SMConvND(2, 4, 4, kernel_size=3, padding=1)
        with torch.no_grad():
            if collapse == "scales":
                conv.scales.zero_()
            else:
                for parameter in conv.parameters():
                    parameter.zero_()
        out = conv(torch.randn(1, 4, 8, 8), torch.zeros(1, 4))
        assert not torch.isnan(out).any()


class TestSMConvNDPlacement:
    """The modulation parameters live wherever the weight was asked to live."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
    def test_an_explicit_dtype_reaches_every_parameter(self, dtype):
        """Test that the scales and gain are built in the requested type."""
        conv = SMConvND(2, 4, kernel_size=3, dtype=dtype)
        assert conv.weight.dtype == dtype
        assert conv.scales.dtype == dtype
        assert conv.gain.dtype == dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_an_explicit_device_reaches_every_parameter(self):
        """Test that a module asked for the accelerator is not left split."""
        conv = SMConvND(2, 4, kernel_size=3, device="cuda")
        assert conv.weight.is_cuda
        assert conv.scales.is_cuda
        assert conv.gain.is_cuda
        out = conv(torch.randn(1, 4, 8, 8, device="cuda"))
        assert out.is_cuda


class TestSMConvBlock:
    """Tests for the block wrapping the self-modulated convolution."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_shapes(self, dimensions):
        """Test that the block maps between the requested channel counts."""
        block = SMConvBlock(dimensions, 4, 6)
        out = block(torch.randn(2, 4, *([8] * dimensions)))
        assert out.shape == (2, 6, *([8] * dimensions))

    def test_normalization_arguments_are_accepted_and_ignored(self):
        """Test that the block stands in for one that takes normalization arguments."""
        block = SMConvBlock(2, 4, 4, norm_type="group", num_groups=2)
        assert block(torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8)
        assert not any("Norm" in type(m).__name__ for m in block.modules())

    def test_the_activation_is_optional(self):
        """Test that the block can be built without an activation."""
        assert SMConvBlock(2, 4, 4, act_fn=None).act is None

    def test_the_registries_carry_the_block(self):
        """Test that the block is reachable by name, like its normalized sibling."""
        assert CONV_BLOCK_MAP["SMConvBlock"] is SMConvBlock
        assert "SMConvBlock" in get_args(ConvBlockTypes)


class TestSMConvResidualBlock:
    """Tests for the residual block built on self-modulated convolutions."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_shapes(self, dimensions):
        """Test that the block maps between the requested channel counts."""
        block = SMConvResidualBlock(dimensions, 6, 8, res_groups=2)
        out = block(torch.randn(2, 6, *([8] * dimensions)))
        assert out.shape == (2, 8, *([8] * dimensions))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_normalizations_are_replaced(self, dimensions):
        """Test that no normalization survives the substitution."""
        block = SMConvResidualBlock(dimensions, 4, 4, res_groups=2)
        assert isinstance(block.norm1, torch.nn.Identity)
        assert isinstance(block.norm2, torch.nn.Identity)
        assert isinstance(block.conv1, SMConvND)
        assert isinstance(block.conv2, SMConvND)
        assert not any("Norm" in type(m).__name__ for m in block.modules())

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_flow(self, dimensions):
        """Test that the block trains end to end."""
        block = SMConvResidualBlock(dimensions, 4, 4, res_groups=2)
        block(torch.randn(2, 4, *([8] * dimensions))).pow(2).sum().backward()
        assert all(p.grad is not None for p in block.parameters())

    def test_an_additive_time_embedding_still_works(self):
        """Test that the block keeps the time injection it can support."""
        block = SMConvResidualBlock(2, 4, 4, True, 8, res_groups=2)
        out = block(torch.randn(2, 4, 8, 8), torch.randn(2, 8))
        assert out.shape == (2, 4, 8, 8)

    def test_a_modulating_time_embedding_raises(self):
        """Test that the block rejects the injection its convolution replaces."""
        with pytest.raises(ValueError, match="scale-shift time injection"):
            SMConvResidualBlock(
                2, 4, 4, True, 8, res_groups=2, res_time_injection="scale_shift"
            )

    def test_the_registries_carry_the_block(self):
        """Test that the block is reachable by name."""
        assert RESIDUAL_BLOCK_MAP["SMConvResidualBlock"] is SMConvResidualBlock
        assert "SMConvResidualBlock" in get_args(ResidualBlockTypes)
        assert set(get_args(ResidualBlockTypes)) <= set(RESIDUAL_BLOCK_MAP)

    @pytest.mark.parametrize(
        "name", ["SMConvAutoencoderUpBlock", "SMConvAutoencoderMidBlock"]
    )
    def test_the_autoencoder_wrappers_are_registered(self, name):
        """Test that the blocks a decoder selects by name exist."""
        assert name in BLOCK_MAP

    def test_the_up_block_wrapper_builds_and_runs(self):
        """Test the decoder block bound to the self-modulated residual block."""
        block = BLOCK_MAP["SMConvAutoencoderUpBlock"](2, 8, 4, res_args={"res_groups": 2})
        assert block(torch.randn(1, 8, 8, 8)).shape == (1, 4, 8, 8)
        assert not any("Norm" in type(m).__name__ for m in block.modules())

    def test_the_mid_block_wrapper_builds_and_runs(self):
        """Test the bottleneck block bound to the self-modulated residual block."""
        block = BLOCK_MAP["SMConvAutoencoderMidBlock"](
            2, channels=8, res_args={"res_groups": 2}
        )
        assert block(torch.randn(1, 8, 8, 8)).shape == (1, 8, 8, 8)


class TestGaussianBlurND:
    """Tests for the separable Gaussian blur."""

    @pytest.mark.parametrize("kernel_size", [3, 4, 5, 9])
    def test_the_kernel_is_normalized_and_symmetric(self, kernel_size):
        """Test the one-dimensional kernel the blur is built from."""
        kernel = gaussian_kernel_1d(kernel_size, 1.5)
        assert kernel.shape == (kernel_size,)
        assert float(kernel.sum()) == pytest.approx(1.0)
        assert torch.allclose(kernel, kernel.flip(0))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_shape_is_preserved(self, dimensions):
        """Test that the blur does not change the resolution."""
        blur = GaussianBlurND(dimensions, 5, 1.0)
        x = torch.randn(2, 3, *([16] * dimensions))
        assert blur(x).shape == x.shape

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_a_constant_field_survives(self, dimensions):
        """Test that a normalized blur leaves a constant untouched."""
        blur = GaussianBlurND(dimensions, 5, 1.0)
        x = torch.full((1, 2, *([12] * dimensions)), 3.0, dtype=torch.float64)
        assert torch.allclose(blur(x), x, atol=1e-9)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_blur_removes_high_frequencies(self, dimensions):
        """Test that the blur reduces the variance of white noise."""
        blur = GaussianBlurND(dimensions, 5, 1.0)
        x = torch.randn(4, 2, *([16] * dimensions))
        assert float(blur(x).var()) < float(x.var())

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_channels_do_not_mix(self, dimensions):
        """Test that the blur is depthwise."""
        blur = GaussianBlurND(dimensions, 5, 1.0)
        x = torch.zeros(1, 2, *([9] * dimensions))
        x[:, 0] = 1.0
        out = blur(x)
        assert float(out[:, 1].abs().max()) == pytest.approx(0.0)
        assert float(out[:, 0].abs().max()) > 0

    def test_the_kernel_is_not_persisted(self):
        """Test that the derived kernel stays out of the checkpoint."""
        blur = GaussianBlurND(2, 5, 1.0)
        assert dict(blur.state_dict()) == {}
        assert len(list(blur.buffers())) == 1

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_blur_follows_the_input_dtype(self, dtype):
        """Test that the kernel is cast to whatever the input carries."""
        out = GaussianBlurND(2, 5, 1.0)(torch.randn(1, 1, 8, 8, dtype=dtype))
        assert out.dtype is dtype

    def test_gradients_flow_through_the_blur(self):
        """Test that the blur is differentiable, as the losses need."""
        x = torch.randn(1, 2, 8, 8, requires_grad=True)
        GaussianBlurND(2, 5, 1.0)(x).pow(2).sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0

    def test_repr_names_the_configuration(self):
        """Test that the representation carries the identifying fields."""
        assert "kernel_size=5" in repr(GaussianBlurND(2, 5, 1.0))

    def test_a_degenerate_kernel_raises(self):
        """Test that the kernel arguments are validated."""
        with pytest.raises(ValueError, match="at least one tap"):
            gaussian_kernel_1d(0, 1.0)
        with pytest.raises(ValueError, match="must be positive"):
            gaussian_kernel_1d(5, 0.0)

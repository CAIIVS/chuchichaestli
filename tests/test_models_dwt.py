# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the wavelet transform layers."""

import pickle
from typing import get_args

import pytest
import torch

from chuchichaestli.dwt.functional import dwt_coeff_len, dwtn
from chuchichaestli.dwt.wavelet import Wavelet
from chuchichaestli.models.downsampling import (
    DOWNSAMPLE_FUNCTIONS,
    DownsampleTypes,
    DownsampleWavelet,
)
from chuchichaestli.models.dwt import (
    WAVELET_LAYER_MAP,
    InverseWaveletTransform2D,
    InverseWaveletTransformND,
    LowpassWaveletTransform2D,
    LowpassWaveletTransformND,
    MultilevelWaveletTransformND,
    WaveletTransform1D,
    WaveletTransform2D,
    WaveletTransform3D,
    WaveletTransformND,
    dwt_nd,
    idwt_nd,
    subband_names,
    wavedec_nd,
    waverec_nd,
)
from chuchichaestli.models.resample import ChannelResample
from chuchichaestli.models.upsampling import (
    UPSAMPLE_FUNCTIONS,
    UpsampleTypes,
    UpsampleWavelet,
)


DIMENSIONS = [1, 2, 3]
ORDERS = ["subband", "channel"]
WAVELETS = ["haar", "db2", "bior2.2"]


class TestWaveletTransformND:
    """Tests for the single-level transform layer."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    @pytest.mark.parametrize("order", ORDERS)
    def test_channels_grow_and_the_resolution_halves(self, dimensions, order):
        """Test that the subbands move into the channel axis."""
        layer = WaveletTransformND(dimensions, "haar", "periodization", order)
        x = torch.randn(2, 3, *([16] * dimensions))
        out = layer(x)
        assert out.shape == (2, 3 * 2**dimensions, *([8] * dimensions))
        assert layer.factor == 2
        assert layer.changes_channels

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("name", WAVELETS)
    def test_the_inverse_layer_reconstructs(self, dimensions, order, name):
        """Test that the two layers compose to the identity."""
        shape = tuple(8 + 2 * i for i in range(dimensions))
        forward = WaveletTransformND(dimensions, name, "symmetric", order)
        inverse = InverseWaveletTransformND(dimensions, name, "symmetric", order)
        x = torch.randn(2, 3, *shape, dtype=torch.float64)
        assert torch.allclose(inverse(forward(x), output_size=shape), x, atol=1e-9)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_two_orders_are_a_permutation_of_each_other(self, dimensions):
        """Test that the layouts differ only in how the channel axis is read."""
        x = torch.randn(2, 3, *([8] * dimensions), dtype=torch.float64)
        by_subband = WaveletTransformND(dimensions, "db2", "zero", "subband")(x)
        by_channel = WaveletTransformND(dimensions, "db2", "zero", "channel")(x)
        count = 2**dimensions
        regrouped = by_channel.unflatten(1, (3, count)).transpose(1, 2).flatten(1, 2)
        assert torch.allclose(regrouped, by_subband)

    def test_haar_is_the_two_dimensional_butterfly(self):
        """Test the one case whose subbands can be written down by hand."""
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
        out = WaveletTransformND(2, "haar", "periodization")(x).flatten()
        assert out.tolist() == pytest.approx([5.0, -1.0, -2.0, 0.0])

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_layer_holds_no_state(self, dimensions):
        """Test that the filters are constants rather than parameters or buffers."""
        layer = WaveletTransformND(dimensions, "db4")
        assert list(layer.parameters()) == []
        assert list(layer.buffers()) == []
        assert dict(layer.state_dict()) == {}

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_layer_follows_the_input_dtype(self, dtype):
        """Test that the filters are derived for whatever the input carries."""
        out = WaveletTransformND(2, "db2")(torch.randn(1, 1, 8, 8, dtype=dtype))
        assert out.dtype is dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_the_layer_follows_the_input_device(self):
        """Test that the transform runs wherever the input lives."""
        layer = WaveletTransformND(2, "db2")
        out = layer(torch.randn(1, 1, 8, 8, device="cuda"))
        assert out.device.type == "cuda"

    def test_pickling_round_trips(self):
        """Test that a layer survives being sent to a worker process."""
        layer = WaveletTransformND(2, "db2", "symmetric")
        clone = pickle.loads(pickle.dumps(layer))
        x = torch.randn(1, 2, 8, 8, dtype=torch.float64)
        assert torch.allclose(clone(x), layer(x))

    def test_repr_names_the_configuration(self):
        """Test that the representation carries the identifying fields."""
        text = repr(WaveletTransformND(2, "db2", "symmetric"))
        assert "db2" in text and "symmetric" in text and "dimensions=2" in text

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_a_wrong_rank_raises(self, dimensions):
        """Test that the layer checks the rank of its input."""
        layer = WaveletTransformND(dimensions, "haar")
        with pytest.raises(ValueError, match="dimensional input"):
            layer(torch.randn(*([4] * (dimensions + 3))))

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_reach_the_input(self, dimensions):
        """Test that the layer is differentiable, as the losses need."""
        x = torch.randn(1, 2, *([8] * dimensions), requires_grad=True)
        WaveletTransformND(dimensions, "db2").forward(x).pow(2).sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0


class TestLowpassWaveletTransformND:
    """Tests for the approximation-only layer."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    @pytest.mark.parametrize("name", WAVELETS)
    def test_it_equals_the_approximation_band_of_the_full_transform(
        self, dimensions, name
    ):
        """Test that skipping the detail bands changes nothing but the cost."""
        x = torch.randn(2, 3, *([8] * dimensions), dtype=torch.float64)
        lowpass = LowpassWaveletTransformND(dimensions, name, "symmetric")(x)
        axes = tuple(range(-dimensions, 0))
        full = dwtn(x, name, "symmetric", axes)["a" * dimensions]
        assert torch.allclose(lowpass, full, atol=1e-12)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_channel_count_is_untouched(self, dimensions):
        """Test that the approximation branch keeps the channels it was given."""
        layer = LowpassWaveletTransformND(dimensions, "haar", "periodization")
        out = layer(torch.randn(2, 5, *([16] * dimensions)))
        assert out.shape == (2, 5, *([8] * dimensions))
        assert not layer.changes_channels


class TestMultilevelWaveletTransformND:
    """Tests for the recursive multi-level layer."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    @pytest.mark.parametrize("levels", [1, 2, 3])
    def test_each_level_halves_again(self, dimensions, levels):
        """Test the resolution and channel count of every level."""
        layer = MultilevelWaveletTransformND(dimensions, "haar", "periodization", levels=levels)
        out = layer(torch.randn(2, 3, *([32] * dimensions)))
        assert len(out) == levels
        assert layer.factor == 2**levels
        for level, band in enumerate(out, start=1):
            assert band.shape == (
                2,
                3 * 2**dimensions,
                *([32 // 2**level] * dimensions),
            )

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_next_level_decomposes_the_approximation(self, order):
        """Test that the recursion runs on the approximation band."""
        layer = MultilevelWaveletTransformND(2, "haar", "periodization", order, levels=2)
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        first, second = layer(x)
        level_one = dwt_nd(x, 2, "haar", "periodization", order)
        if order == "subband":
            approx = level_one.chunk(4, dim=1)[0]
        else:
            approx = level_one.unflatten(1, (3, 4)).unbind(dim=2)[0]
        assert torch.allclose(first, level_one)
        assert torch.allclose(
            second, dwt_nd(approx, 2, "haar", "periodization", order)
        )

    def test_a_non_positive_level_count_raises(self):
        """Test that a decomposition needs at least one level."""
        with pytest.raises(ValueError, match="at least one level"):
            MultilevelWaveletTransformND(2, "haar", levels=0)

    def test_repr_reports_the_levels(self):
        """Test that the representation carries the level count."""
        assert "levels=3" in repr(MultilevelWaveletTransformND(2, "haar", levels=3))


class TestNamedSubbandLayers:
    """Tests for the dimension-fixed layers that return the subbands separately."""

    @pytest.mark.parametrize(
        "layer_cls,dimensions",
        [(WaveletTransform1D, 1), (WaveletTransform2D, 2), (WaveletTransform3D, 3)],
    )
    def test_one_tensor_per_subband(self, layer_cls, dimensions):
        """Test that the transform yields the subbands separately."""
        layer = layer_cls(wavelet="haar", mode="periodization")
        bands = layer(torch.randn(2, 3, *([16] * dimensions)))
        assert len(bands) == 2**dimensions
        assert layer.subbands == subband_names(dimensions)
        assert all(b.shape == (2, 3, *([8] * dimensions)) for b in bands)

    def test_the_two_dimensional_order_is_the_classical_one(self):
        """Test that the subbands come out as `LL`, `LH`, `HL`, `HH`."""
        layer = WaveletTransform2D(wavelet="db2", mode="symmetric")
        x = torch.randn(1, 2, 12, 12, dtype=torch.float64)
        bands = dict(zip(layer.subbands, layer(x), strict=True))
        reference = dwtn(x, "db2", "symmetric", (-2, -1))
        assert set(bands) == {"aa", "ad", "da", "dd"}
        for key, band in bands.items():
            assert torch.allclose(band, reference[key], atol=1e-12)

    def test_the_named_inverse_reconstructs(self):
        """Test that the separated subbands feed back into the inverse."""
        forward = WaveletTransform2D(wavelet="db2", mode="symmetric")
        inverse = InverseWaveletTransform2D(wavelet="db2", mode="symmetric")
        x = torch.randn(2, 3, 12, 12, dtype=torch.float64)
        out = inverse(*forward(x), output_size=(12, 12))
        assert torch.allclose(out, x, atol=1e-9)

    def test_the_inverse_reports_the_subbands_it_expects(self):
        """Test that the inverse names the subbands it takes, in order."""
        assert InverseWaveletTransform2D(wavelet="haar").subbands == ("aa", "ad", "da", "dd")

    def test_the_wrong_number_of_subbands_raises(self):
        """Test that the inverse checks how many subbands it was handed."""
        inverse = InverseWaveletTransform2D(wavelet="haar")
        with pytest.raises(ValueError, match="4 subbands"):
            inverse(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))

    def test_the_lowpass_variants_keep_their_channels(self):
        """Test the dimension-fixed approximation layers."""
        assert LowpassWaveletTransform2D(wavelet="haar", mode="periodization")(
            torch.randn(1, 3, 8, 8)
        ).shape == (1, 3, 4, 4)

    def test_the_registry_lists_the_layers(self):
        """Test the name to class map."""
        assert WAVELET_LAYER_MAP["WaveletTransformND"] is WaveletTransformND
        assert set(WAVELET_LAYER_MAP) == {
            "WaveletTransformND",
            "InverseWaveletTransformND",
            "LowpassWaveletTransformND",
            "MultilevelWaveletTransformND",
        }


class TestBatchedFunctions:
    """Tests for the batched functional wrappers."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    @pytest.mark.parametrize("order", ORDERS)
    @pytest.mark.parametrize("name", WAVELETS)
    def test_multi_level_round_trip(self, dimensions, order, name):
        """Test that a decomposition and its reconstruction compose to the identity."""
        shape = tuple([16] * dimensions)
        x = torch.randn(2, 3, *shape, dtype=torch.float64)
        levels = 2
        coeffs = wavedec_nd(x, dimensions, name, "symmetric", levels, order)
        filter_len = Wavelet.from_name(name).dec_len
        sizes, current = [], list(shape)
        for _ in range(levels):
            sizes.append(tuple(current))
            current = [dwt_coeff_len(n, filter_len, "symmetric") for n in current]
        out = waverec_nd(coeffs, dimensions, name, "symmetric", order, sizes[::-1])
        assert torch.allclose(out, x, atol=1e-8)

    def test_an_unknown_subband_order_raises(self):
        """Test that the subband order is validated."""
        with pytest.raises(ValueError, match="Unsupported subband order"):
            dwt_nd(torch.randn(1, 1, 8, 8), 2, "haar", "zero", "nope")

    def test_an_unknown_subband_order_raises_on_the_inverse(self):
        """Test that the inverse validates the subband order too."""
        with pytest.raises(ValueError, match="Unsupported subband order"):
            idwt_nd(torch.randn(1, 4, 4, 4), 2, "haar", "zero", "nope")

    def test_a_channel_count_that_does_not_divide_raises(self):
        """Test that the inverse checks the channel count against the subbands."""
        with pytest.raises(ValueError, match="multiple of 4"):
            idwt_nd(torch.randn(1, 3, 4, 4), 2, "haar")

    def test_a_non_positive_level_count_raises(self):
        """Test that a decomposition needs at least one level."""
        with pytest.raises(ValueError, match="at least one level"):
            wavedec_nd(torch.randn(1, 1, 8, 8), 2, "haar", "zero", 0)

    def test_empty_coefficients_raise(self):
        """Test that a reconstruction needs at least one level."""
        with pytest.raises(ValueError, match="at least one level"):
            waverec_nd([], 2, "haar")


class TestSamplingBlocks:
    """Tests for the wavelet and channel-only sampling blocks."""

    def test_the_registries_match_the_type_literals(self):
        """Test that every advertised sampler name is registered."""
        assert set(get_args(DownsampleTypes)) <= set(DOWNSAMPLE_FUNCTIONS)
        assert set(get_args(UpsampleTypes)) <= set(UPSAMPLE_FUNCTIONS)
        assert DOWNSAMPLE_FUNCTIONS["DownsampleWavelet"] is DownsampleWavelet
        assert UPSAMPLE_FUNCTIONS["UpsampleWavelet"] is UpsampleWavelet

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_channel_sampler_keeps_the_resolution(self, dimensions):
        """Test that channel resampling scales by a factor of one."""
        shape = tuple([8] * dimensions)
        widen = ChannelResample(dimensions, 4, 12)
        narrow = ChannelResample(dimensions, 12, 4)
        x = torch.randn(2, 4, *shape)
        assert widen.factor == narrow.factor == 1
        assert widen(x).shape == (2, 12, *shape)
        assert narrow(widen(x)).shape == (2, 4, *shape)

    def test_channel_resampling_serves_both_directions(self):
        """Test that one class is registered as a sampler either way."""
        assert DOWNSAMPLE_FUNCTIONS["ChannelResample"] is ChannelResample
        assert UPSAMPLE_FUNCTIONS["ChannelResample"] is ChannelResample

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_the_wavelet_sampler_halves_and_doubles(self, dimensions):
        """Test the spatial factor of the wavelet samplers."""
        shape = tuple([16] * dimensions)
        widened = 4 * 2**dimensions
        down = DownsampleWavelet(dimensions, 4, widened)
        up = UpsampleWavelet(dimensions, widened, 4)
        x = torch.randn(2, 4, *shape)
        assert down.factor == up.factor == 2
        assert down(x).shape == (2, widened, *([8] * dimensions))
        assert up(down(x)).shape == (2, 4, *shape)

    @pytest.mark.parametrize(
        "cls", [ChannelResample, DownsampleWavelet, UpsampleWavelet]
    )
    def test_every_new_sampler_changes_channels(self, cls):
        """Test the attribute the encoder and U-Net read to size their levels."""
        assert cls.changes_channels

    def test_a_factor_other_than_two_raises(self):
        """Test that the wavelet samplers only halve or double."""
        with pytest.raises(ValueError, match="only a factor of two"):
            DownsampleWavelet(2, 4, 16, factor=4)
        with pytest.raises(ValueError, match="only a factor of two"):
            UpsampleWavelet(2, 16, 4, factor=4)

    def test_channel_counts_that_do_not_fit_raise(self):
        """Test the channel bookkeeping of the wavelet samplers."""
        with pytest.raises(ValueError, match="Cannot decompose"):
            DownsampleWavelet(2, 4, 6)
        with pytest.raises(ValueError, match="Cannot synthesize"):
            UpsampleWavelet(2, 7, 4)

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_gradients_flow_through_the_samplers(self, dimensions):
        """Test that the samplers are trainable."""
        down = DownsampleWavelet(dimensions, 4, 4 * 2**dimensions)
        x = torch.randn(2, 4, *([8] * dimensions), requires_grad=True)
        down(x).pow(2).sum().backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in down.parameters())


class TestConstantResolutionUNet:
    """Tests for the U-Net the channel-only samplers turn into a flat one."""

    @pytest.mark.parametrize("dimensions", DIMENSIONS)
    def test_a_unet_keeps_its_resolution(self, dimensions):
        """Test that channel-only sampling leaves the spatial axes alone."""
        from chuchichaestli.models.unet import UNet

        channels = 2**dimensions
        net = UNet(
            dimensions=dimensions,
            in_channels=channels,
            n_channels=8,
            out_channels=channels,
            down_block_types=("DownBlock",) * 3,
            mid_block_type="MidBlock",
            up_block_types=("UpBlock",) * 3,
            block_out_channel_mults=(1, 2, 3),
            num_blocks_per_level=1,
            downsample_type="ChannelResample",
            upsample_type="ChannelResample",
            time_embedding=None,
            groups=4,
            res_groups=4,
        )
        x = torch.randn(1, channels, *([8] * dimensions))
        assert net(x).shape == x.shape

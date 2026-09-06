# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the separable N-dimensional discrete wavelet transform."""

import pytest
import torch

from chuchichaestli.dwt.functional import (
    dwt,
    dwt_coeff_len,
    dwt_max_level,
    dwtn,
    idwt,
    idwtn,
    subband_keys,
    wavedec,
    wavedecn,
    waverec,
    waverecn,
)
from chuchichaestli.dwt.modes import MODE_TO_CODE
from chuchichaestli.dwt.wavelet import Wavelet


MODES = sorted(MODE_TO_CODE)
WAVELETS = ["haar", "db2", "db4", "sym4", "coif1", "bior2.2", "rbio3.3"]
CIRCULAR = ("periodic", "periodization")

# One level of `db2` over `arange(1, 13)`, per extension mode.
GOLDEN_DWT = {
    "zero": (
        (
            -0.034675177060507, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            15.350787689443571,
        ),
        (
            -0.12940952255126, 0.0, 0.0, 0.0, 0.0, 0.0, -4.113231164568025,
        ),
    ),
    "constant": (
        (
            1.284804039821835, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            16.48759983533261,
        ),
        (
            -0.482962913144534, 0.0, 0.0, 0.0, 0.0, 0.0, 0.129409522551261,
        ),
    ),
    "symmetric": (
        (
            1.767766952966369, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            16.617009357883866,
        ),
        (
            -0.612372435695794, 0.0, 0.0, 0.0, 0.0, 0.0, 0.612372435695796,
        ),
    ),
    "reflect": (
        (
            3.087246169848711, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            16.522275012393116,
        ),
        (
            -0.965925826289068, 0.0, 0.0, 0.0, 0.0, 0.0, 0.258819045102522,
        ),
    ),
    "periodic": (
        (
            15.316112512383064, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            15.316112512383064,
        ),
        (
            -4.242640687119286, 0.0, 0.0, 0.0, 0.0, 0.0, -4.242640687119286,
        ),
    ),
    "periodization": (
        (
            6.692130429902464, 3.725002596914244, 6.553429721660434,
            9.381856846406624, 12.210283971152814, 16.59162536651413,
        ),
        (
            -1.552914270615124, 0.0, 0.0, 0.0, 1e-15, 5.79555495773441,
        ),
    ),
    "antisymmetric": (
        (
            -1.837117307087384, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            14.084566021003274,
        ),
        (
            0.353553390593274, 0.0, 0.0, 0.0, 0.0, 0.0, -8.838834764831846,
        ),
    ),
    "antireflect": (
        (
            -0.517638090205041, 2.310789034541149, 5.139216159287339,
            7.96764328403353, 10.79607040877972, 13.624497533525911,
            16.4529246582721,
        ),
        (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-15,
        ),
    ),
}


def sizes_per_level(shape, filter_len, mode, levels):
    """Spatial shape at each level, finest first, as `waverecn` wants it."""
    out, current = [], list(shape)
    for _ in range(levels):
        out.append(tuple(current))
        current = [dwt_coeff_len(n, filter_len, mode) for n in current]
    return out


class TestGolden:
    """Tests against coefficients taken from PyWavelets."""

    @pytest.mark.parametrize("mode", MODES)
    def test_one_level_matches_published_coefficients(self, mode):
        """Test `db2` over a ramp against PyWavelets, for every extension mode."""
        x = torch.arange(1.0, 13.0, dtype=torch.float64)
        approx, detail = dwt(x, "db2", mode)
        golden_a, golden_d = GOLDEN_DWT[mode]
        assert approx.tolist() == pytest.approx(list(golden_a), abs=1e-12)
        assert detail.tolist() == pytest.approx(list(golden_d), abs=1e-12)

    def test_haar_is_the_sum_and_difference_butterfly(self):
        """Test the one transform whose coefficients can be written down by hand."""
        x = torch.tensor([1.0, 3.0, 5.0, 11.0], dtype=torch.float64)
        approx, detail = dwt(x, "haar", "zero")
        root = 2**0.5
        assert approx.tolist() == pytest.approx([4 / root, 16 / root])
        assert detail.tolist() == pytest.approx([-2 / root, -6 / root])


class TestPerfectReconstruction:
    """Tests for the property the whole transform stands on."""

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("length", [8, 9, 16])
    def test_one_level_one_axis(self, name, mode, length):
        """Test that a single level reconstructs a signal of either parity."""
        x = torch.randn(2, length, dtype=torch.float64)
        approx, detail = dwt(x, name, mode)
        assert torch.allclose(idwt(approx, detail, name, mode, output_size=length), x, atol=1e-9)

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    @pytest.mark.parametrize("odd", [False, True])
    def test_one_level_n_dimensional(self, name, mode, dimensions, odd):
        """Test that a single level reconstructs for one, two and three axes."""
        shape = tuple(6 + odd + i for i in range(dimensions))
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, 3, *shape, dtype=torch.float64)
        bands = dwtn(x, name, mode, axes)
        out = idwtn(bands, name, mode, axes, output_size=shape)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-9)

    @pytest.mark.parametrize("name", ["haar", "db3", "bior2.2"])
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_multi_level(self, name, mode, dimensions):
        """Test that a three-level decomposition reconstructs exactly."""
        shape = tuple(16 + 2 * i for i in range(dimensions))
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, *shape, dtype=torch.float64)
        filter_len = Wavelet.from_name(name).dec_len
        coeffs = wavedecn(x, name, mode, 3, axes)
        sizes = sizes_per_level(shape, filter_len, mode, 3)[::-1]
        out = waverecn(coeffs, name, mode, axes, output_size=sizes)
        assert torch.allclose(out, x, atol=1e-8)

    @pytest.mark.parametrize("name", ["haar", "db2"])
    @pytest.mark.parametrize("mode", MODES)
    def test_multi_level_one_axis(self, name, mode):
        """Test the single-axis multi-level helpers."""
        x = torch.randn(3, 32, dtype=torch.float64)
        filter_len = Wavelet.from_name(name).dec_len
        coeffs = wavedec(x, name, mode, 3)
        sizes = [s[0] for s in sizes_per_level((32,), filter_len, mode, 3)][::-1]
        assert torch.allclose(waverec(coeffs, name, mode, output_size=sizes), x, atol=1e-8)

    @pytest.mark.parametrize("name", ["haar", "db2", "bior2.2"])
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("length", [17, 30, 31])
    def test_multi_level_one_axis_without_recorded_sizes(self, name, mode, length):
        """Test that an odd level length inverts without an `output_size`."""
        x = torch.randn(3, length, dtype=torch.float64)
        coeffs = wavedec(x, name, mode, 3)
        out = waverec(coeffs, name, mode)
        assert out.shape[-1] >= length
        assert torch.allclose(out[..., :length], x, atol=1e-8)

    @pytest.mark.parametrize("name", ["haar", "db3"])
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dimensions", [1, 2])
    def test_multi_level_without_recorded_sizes(self, name, mode, dimensions):
        """Test that odd axes invert when no `output_size` was recorded."""
        shape = tuple(17 + 2 * i for i in range(dimensions))
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, *shape, dtype=torch.float64)
        coeffs = wavedecn(x, name, mode, 3, axes)
        out = waverecn(coeffs, name, mode, axes)
        crop = (slice(None),) + tuple(slice(0, n) for n in shape)
        assert torch.allclose(out[crop], x, atol=1e-8)

    def test_a_decomposition_of_no_levels_round_trips_trivially(self):
        """Test the degenerate decomposition with nothing to invert."""
        x = torch.randn(4, 8, dtype=torch.float64)
        coeffs = wavedecn(x, "haar", "zero", 0)
        assert torch.allclose(waverecn(coeffs, "haar", "zero"), x)


class TestCoefficientLengths:
    """Tests for the coefficient bookkeeping."""

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("length", [8, 9, 12, 31])
    def test_predicted_length_matches_the_transform(self, name, mode, length):
        """Test that `dwt_coeff_len` predicts what the transform produces."""
        filter_len = Wavelet.from_name(name).dec_len
        approx, _ = dwt(torch.randn(length, dtype=torch.float64), name, mode)
        assert approx.shape[0] == dwt_coeff_len(length, filter_len, mode)

    @pytest.mark.parametrize("mode", CIRCULAR)
    def test_periodization_is_critically_sampled(self, mode):
        """Test that periodization keeps the coefficient count at half the input."""
        assert dwt_coeff_len(16, 8, "periodization") == 8
        assert dwt_coeff_len(15, 8, "periodization") == 8

    def test_redundant_modes_grow_with_the_filter(self):
        """Test that the other modes produce more coefficients for longer filters."""
        assert dwt_coeff_len(16, 2, "zero") == 8
        assert dwt_coeff_len(16, 8, "zero") == 11

    def test_max_level_shrinks_with_the_filter(self):
        """Test the maximum useful number of levels."""
        assert dwt_max_level(32, 2) == 5
        assert dwt_max_level(1, 8) == 0
        assert dwt_max_level(8, 20) == 0

    def test_a_non_positive_length_raises(self):
        """Test that empty signals and filters are rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            dwt_coeff_len(0, 4, "zero")


class TestSubbands:
    """Tests for the naming and ordering of the subbands."""

    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_keys_run_over_every_branch_combination(self, dimensions):
        """Test that the keys enumerate `a`/`d` per axis, first axis most significant."""
        keys = subband_keys(dimensions)
        assert len(keys) == 2**dimensions
        assert len(set(keys)) == len(keys)
        assert all(len(k) == dimensions and set(k) <= {"a", "d"} for k in keys)
        assert keys[0] == "a" * dimensions
        assert keys[-1] == "d" * dimensions

    def test_two_dimensional_keys_are_ordered_like_pywavelets(self):
        """Test the two-dimensional ordering the image-processing names assume."""
        assert subband_keys(2) == ("aa", "ad", "da", "dd")

    def test_the_transform_produces_exactly_those_keys(self):
        """Test that the transform's output is keyed by `subband_keys`."""
        bands = dwtn(torch.randn(2, 8, 8), "haar", "zero", (-2, -1))
        assert tuple(bands) == subband_keys(2)

    def test_the_approximation_carries_the_energy(self):
        """Test that a constant image lands entirely in the approximation band."""
        bands = dwtn(torch.ones(1, 8, 8, dtype=torch.float64), "haar", "periodization", (-2, -1))
        assert bands["aa"].abs().sum() > 0
        for key in ("ad", "da", "dd"):
            assert bands[key].abs().max() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ["haar", "db2", "coif1"])
    def test_orthogonal_transforms_preserve_energy(self, name):
        """Test Parseval's identity, which only the critically sampled mode can satisfy."""
        x = torch.randn(2, 16, 16, dtype=torch.float64)
        bands = dwtn(x, name, "periodization", (-2, -1))
        energy = sum(float((b**2).sum()) for b in bands.values())
        assert energy == pytest.approx(float((x**2).sum()), rel=1e-10)


class TestAxes:
    """Tests for selecting which axes are transformed."""

    @pytest.mark.parametrize("axes", [(0,), (1,), (2,), (0, 2), (-3, -1), (0, 1, 2)])
    def test_only_the_requested_axes_are_halved(self, axes):
        """Test the output shape for an arbitrary axis selection."""
        x = torch.randn(8, 10, 12, dtype=torch.float64)
        bands = dwtn(x, "haar", "periodization", axes)
        expected = [n // 2 if i in {a % 3 for a in axes} else n for i, n in enumerate(x.shape)]
        assert list(next(iter(bands.values())).shape) == expected

    @pytest.mark.parametrize("axes", [(0,), (1,), (0, 2), (0, 1, 2)])
    def test_reconstruction_restores_the_original_layout(self, axes):
        """Test that the axes come back in their original positions."""
        x = torch.randn(8, 10, 12, dtype=torch.float64)
        bands = dwtn(x, "db2", "zero", axes)
        shape = tuple(x.shape[a % 3] for a in axes)
        out = idwtn(bands, "db2", "zero", axes, output_size=shape)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-9)

    def test_the_trailing_axis_is_the_default(self):
        """Test that omitting `axes` transforms the last one."""
        x = torch.randn(4, 8, dtype=torch.float64)
        assert torch.allclose(dwtn(x, "haar", "zero")["a"], dwt(x, "haar", "zero")[0])

    def test_an_out_of_range_axis_raises(self):
        """Test that an axis outside the tensor is reported."""
        with pytest.raises(ValueError, match="out of range"):
            dwtn(torch.randn(4, 4), "haar", "zero", (5,))

    def test_repeated_axes_raise(self):
        """Test that each axis may only be transformed once."""
        with pytest.raises(ValueError, match="must be distinct"):
            dwtn(torch.randn(4, 4), "haar", "zero", (0, -2))

    def test_more_than_three_axes_raise(self):
        """Test the limit `torch`'s convolutions impose."""
        with pytest.raises(ValueError, match="one to three axes"):
            dwtn(torch.randn(2, 2, 2, 2), "haar", "zero", (0, 1, 2, 3))


class TestDtypesAndDevices:
    """Tests for how the transform treats dtypes and devices."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_dtype_is_preserved(self, dtype):
        """Test that a floating point input keeps its precision."""
        bands = dwtn(torch.randn(2, 8, 8, dtype=dtype), "db2", "zero", (-2, -1))
        assert all(b.dtype is dtype for b in bands.values())

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
    def test_non_floating_input_is_promoted(self, dtype):
        """Test that integer input is promoted rather than rejected."""
        x = torch.ones(2, 8, 8).to(dtype)
        assert dwtn(x, "haar", "zero", (-2, -1))["aa"].dtype is torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_the_transform_runs_on_the_input_device(self):
        """Test that the filters follow the input onto the accelerator."""
        x = torch.randn(2, 8, 8, dtype=torch.float64, device="cuda")
        bands = dwtn(x, "db2", "symmetric", (-2, -1))
        assert all(b.device.type == "cuda" for b in bands.values())
        out = idwtn(bands, "db2", "symmetric", (-2, -1), output_size=(8, 8))
        assert torch.allclose(out, x, atol=1e-9)


class TestDifferentiability:
    """Tests for the gradients the losses depend on."""

    @pytest.mark.parametrize("name", ["haar", "db2", "bior2.2"])
    @pytest.mark.parametrize("mode", MODES)
    def test_the_adjoint_identity_holds(self, name, mode):
        """Test that the backward pass really is the adjoint of the transform.

        The transform is linear, so `<T x, y>` must equal `<x, T* y>` exactly.
        This catches a backward that is right in the bulk and wrong at the
        boundary, which random probing can miss.
        """
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64, requires_grad=True)
        bands = dwtn(x, name, mode, (-2, -1))
        cotangents = {k: torch.randn_like(v) for k, v in bands.items()}
        lhs = sum(float((bands[k] * cotangents[k]).sum().detach()) for k in bands)
        torch.autograd.backward(list(bands.values()), list(cotangents.values()))
        rhs = float((x.detach() * x.grad).sum())
        assert lhs == pytest.approx(rhs, abs=1e-10)

    @pytest.mark.parametrize("name", ["haar", "db2"])
    @pytest.mark.parametrize("mode", MODES)
    def test_gradcheck_of_the_decomposition(self, name, mode):
        """Test the analysis gradients against finite differences."""
        x = torch.randn(1, 2, 6, 8, dtype=torch.float64, requires_grad=True)

        def run(tensor):
            return tuple(dwtn(tensor, name, mode, (-2, -1)).values())

        assert torch.autograd.gradcheck(run, (x,))

    @pytest.mark.parametrize("name", ["haar", "db2"])
    @pytest.mark.parametrize("mode", MODES)
    def test_gradcheck_of_the_reconstruction(self, name, mode):
        """Test the synthesis gradients against finite differences."""
        keys = subband_keys(2)
        bands = [torch.randn(1, 2, 5, 6, dtype=torch.float64, requires_grad=True) for _ in keys]

        def run(*values):
            return idwtn(dict(zip(keys, values, strict=True)), name, mode, (-2, -1), (6, 8))

        assert torch.autograd.gradcheck(run, tuple(bands))

    @pytest.mark.parametrize("mode", MODES)
    def test_second_order_gradients_work(self, mode):
        """Test double backward, which a gradient penalty would need."""
        x = torch.randn(1, 1, 6, 8, dtype=torch.float64, requires_grad=True)

        def run(tensor):
            return tuple(dwtn(tensor, "db2", mode, (-2, -1)).values())

        assert torch.autograd.gradgradcheck(run, (x,))

    def test_gradients_reach_a_multi_level_decomposition(self):
        """Test that every level of a decomposition contributes a gradient."""
        x = torch.randn(1, 1, 16, 16, dtype=torch.float64, requires_grad=True)
        coeffs = wavedecn(x, "db2", "zero", 3, (-2, -1))
        loss = (coeffs[0] ** 2).sum() + sum(
            (band**2).sum() for level in coeffs[1:] for band in level.values()
        )
        loss.backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0


class TestValidation:
    """Tests for the error paths of the transform."""

    def test_a_missing_subband_raises(self):
        """Test that every subband is needed to reconstruct."""
        bands = dwtn(torch.randn(2, 8, 8), "haar", "zero", (-2, -1))
        bands.pop("dd")
        with pytest.raises(ValueError, match="Missing subband"):
            idwtn(bands, "haar", "zero", (-2, -1))

    def test_subbands_of_different_shapes_raise(self):
        """Test that the subbands must agree in shape."""
        bands = dwtn(torch.randn(2, 8, 8), "haar", "zero", (-2, -1))
        bands["dd"] = bands["dd"][..., :-1]
        with pytest.raises(ValueError, match="same shape"):
            idwtn(bands, "haar", "zero", (-2, -1))

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    def test_an_even_length_is_assumed_without_an_output_size(self, name, mode):
        """Test that omitting `output_size` recovers an even-length signal exactly."""
        x = torch.randn(2, 16, dtype=torch.float64)
        approx, detail = dwt(x, name, mode)
        assert torch.allclose(idwt(approx, detail, name, mode), x, atol=1e-9)

    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_the_assumed_length_covers_every_axis(self, dimensions):
        """Test the default reconstruction shape for a multi-axis transform."""
        shape = tuple(8 + 2 * i for i in range(dimensions))
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, *shape, dtype=torch.float64)
        bands = dwtn(x, "db2", "symmetric", axes)
        assert idwtn(bands, "db2", "symmetric", axes).shape == x.shape

    def test_a_short_output_size_raises(self):
        """Test that `output_size` must cover every transformed axis."""
        bands = dwtn(torch.randn(2, 8, 8), "haar", "zero", (-2, -1))
        with pytest.raises(ValueError, match="one length per transformed axis"):
            idwtn(bands, "haar", "zero", (-2, -1), output_size=(8,))

    def test_a_negative_level_raises(self):
        """Test that the number of levels must not be negative."""
        with pytest.raises(ValueError, match="must not be negative"):
            wavedecn(torch.randn(2, 8), "haar", "zero", -1)

    @pytest.mark.parametrize("recompose", [waverecn, waverec])
    def test_empty_coefficients_raise(self, recompose):
        """Test that reconstruction needs at least the approximation band."""
        with pytest.raises(ValueError, match="at least the approximation"):
            recompose([], "haar", "zero")

    def test_the_default_level_is_derived_from_the_shape(self):
        """Test that omitting `level` decomposes as far as the signal allows."""
        coeffs = wavedecn(torch.randn(64, dtype=torch.float64), "db2", "zero")
        assert len(coeffs) - 1 == dwt_max_level(64, 4)


class TestPyWaveletsParity:
    """Cross-checks against PyWavelets, when it is installed."""

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    def test_one_dimensional_coefficients_agree(self, name, mode):
        """Test one level along a single axis against the reference."""
        pywt = pytest.importorskip("pywt")
        x = torch.randn(17, dtype=torch.float64)
        approx, detail = dwt(x, name, mode)
        ref_a, ref_d = pywt.dwt(x.numpy(), name, mode=mode)
        assert approx.numpy() == pytest.approx(ref_a, abs=1e-12)
        assert detail.numpy() == pytest.approx(ref_d, abs=1e-12)

    @pytest.mark.parametrize("name", ["haar", "db3", "bior2.2"])
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dimensions", [2, 3])
    def test_n_dimensional_coefficients_agree(self, name, mode, dimensions):
        """Test one level over several axes against the reference."""
        pywt = pytest.importorskip("pywt")
        shape = tuple(7 + i for i in range(dimensions))
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, *shape, dtype=torch.float64)
        ours = dwtn(x, name, mode, axes)
        ref = pywt.dwtn(x.numpy(), name, mode=mode, axes=axes)
        assert set(ours) == set(ref)
        for key, band in ours.items():
            assert band.numpy() == pytest.approx(ref[key], abs=1e-12)

    @pytest.mark.parametrize("name", ["haar", "db2"])
    @pytest.mark.parametrize("mode", MODES)
    def test_multi_level_coefficients_agree(self, name, mode):
        """Test a three-level decomposition against the reference."""
        pywt = pytest.importorskip("pywt")
        x = torch.randn(40, dtype=torch.float64)
        ours = wavedec(x, name, mode, 3)
        ref = pywt.wavedec(x.numpy(), name, mode=mode, level=3)
        assert len(ours) == len(ref)
        for band, reference in zip(ours, ref, strict=True):
            assert band.numpy() == pytest.approx(reference, abs=1e-11)

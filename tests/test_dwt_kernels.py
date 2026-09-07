# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the optional compiled wavelet kernels.

The kernels are an accelerator, so most of what they owe is already asserted by
the pure-torch suite, which runs through them whenever the extension is built.
What is left here is the agreement between the two paths, and the fallback that
has to work when the extension is absent.
"""

import itertools

import pytest
import torch

from chuchichaestli.dwt import _ext
from chuchichaestli.dwt import wavelet
from chuchichaestli.dwt.functional import dwtn, idwtn, subband_keys, wavedecn
from chuchichaestli.dwt.modes import MODE_TO_CODE

needs_kernels = pytest.mark.skipif(
    not _ext.kernels_built(), reason="the extension is not built"
)
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)

MODES = sorted(MODE_TO_CODE)
WAVELETS = ["haar", "db2", "db4", "bior2.2"]


class TestFallback:
    """Tests for the path taken when the extension is absent or switched off."""

    def test_the_guard_reports_what_is_available(self):
        """Test that availability is about the extension, not about a device."""
        assert _ext.kernels_available() is (
            _ext.kernels_built() and _ext.USE_CUSTOM_KERNELS
        )

    def test_the_switch_forces_the_torch_path(self, monkeypatch):
        """Test that the kernels can be turned off without uninstalling them."""
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        assert _ext.kernels_available() is False

    def test_the_transform_runs_either_way(self, monkeypatch):
        """Test that switching the kernels off changes nothing but the path."""
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        with_kernels = dwtn(x, "db2", "symmetric", (-2, -1))
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        without = dwtn(x, "db2", "symmetric", (-2, -1))
        assert set(with_kernels) == set(without)
        for key, band in with_kernels.items():
            assert torch.allclose(band, without[key], atol=1e-11)


@needs_kernels
class TestAgreement:
    """Tests that the compiled path matches the pure-torch one."""

    @pytest.mark.parametrize(
        "dtype,tol",
        [
            (torch.float64, 1e-12),
            (torch.float32, 1e-5),
            (torch.float16, 2e-2),
            (torch.bfloat16, 1e-1),
            (torch.complex64, 1e-5),
            (torch.complex128, 1e-12),
        ],
    )
    @pytest.mark.parametrize("name", ["haar", "db2", "db4"])
    def test_every_dtype_agrees_with_the_torch_path(
        self, dtype, tol, name, monkeypatch
    ):
        """Test that the kernels serve the reduced and complex types too."""
        assert _ext.kernels_available(torch.device("cpu"), dtype)
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64).to(dtype)
        compiled = dwtn(x, name, "symmetric", (-2, -1))
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        fallback = dwtn(x, name, "symmetric", (-2, -1))
        for key in subband_keys(2):
            assert compiled[key].dtype == dtype
            assert torch.allclose(
                compiled[key].to(torch.complex128),
                fallback[key].to(torch.complex128),
                atol=tol,
                rtol=tol,
            )

    @needs_gpu
    @pytest.mark.parametrize(
        "dtype,tol",
        [
            (torch.float64, 1e-12),
            (torch.float32, 1e-5),
            (torch.float16, 2e-2),
            (torch.bfloat16, 1e-1),
            (torch.complex64, 1e-5),
            (torch.complex128, 1e-12),
        ],
    )
    def test_every_dtype_agrees_on_the_gpu(self, dtype, tol, monkeypatch):
        """Test that the device does not narrow the types the kernels serve."""
        device = torch.device("cuda")
        if not _ext.kernels_available(device):
            pytest.skip("the extension carries no GPU kernels")
        assert _ext.kernels_available(device, dtype)
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64).to(dtype).to(device)
        compiled = dwtn(x, "db2", "symmetric", (-2, -1))
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        fallback = dwtn(x, "db2", "symmetric", (-2, -1))
        for key in subband_keys(2):
            assert compiled[key].dtype == dtype
            assert torch.allclose(
                compiled[key].to(torch.complex128),
                fallback[key].to(torch.complex128),
                atol=tol,
                rtol=tol,
            )

    @needs_kernels
    def test_the_reported_types_are_the_ones_that_work(self):
        """Test that the fallback gate matches what the kernels really dispatch."""
        served = set()
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex32,
            torch.complex64,
            torch.complex128,
        ):
            x = torch.zeros(1, 1, 8, 8).to(dtype)
            filt = torch.zeros(2).to(dtype)
            try:
                _ext._dwt_kernels.dwt_axis(x, filt, filt, 0, 0, 0, 4)
            except (NotImplementedError, RuntimeError):
                continue
            served.add(dtype)
        assert served == _ext.kernel_dtypes()

    @needs_gpu
    @pytest.mark.parametrize("name", ["haar", "db2", "db4", "bior2.2"])
    @pytest.mark.parametrize("mode", MODES)
    def test_the_reconstruction_kernel_matches_the_transposed_convolution(
        self, name, mode, monkeypatch
    ):
        """Test the compiled inverse against the one it stands in for."""
        device = torch.device("cuda")
        if not _ext.kernels_available(device):
            pytest.skip("the extension carries no GPU kernels")
        x = torch.randn(2, 2, 8, 8, dtype=torch.float64, device=device)
        bands = dwtn(x, name, mode, (-2, -1))
        compiled = idwtn(bands, name, mode, (-2, -1), output_size=(8, 8))
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        fallback = idwtn(bands, name, mode, (-2, -1), output_size=(8, 8))
        assert torch.allclose(compiled, fallback, atol=1e-11)

    @needs_gpu
    @pytest.mark.parametrize("name", ["haar", "db2"])
    @pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
    def test_the_reconstruction_kernel_carries_a_gradient(
        self, name, mode, monkeypatch
    ):
        """Test that the compiled inverse agrees with the torch path on gradients."""
        device = torch.device("cuda")
        if not _ext.kernels_available(device):
            pytest.skip("the extension carries no GPU kernels")
        base = torch.randn(2, 2, 8, 8, dtype=torch.float64)
        grads = []
        for use in (False, True):
            monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", use)
            x = base.clone().to(device).requires_grad_(True)
            out = idwtn(dwtn(x, name, mode, (-2, -1)), name, mode, (-2, -1),
                        output_size=(8, 8))
            (out**2).sum().backward()
            grads.append(x.grad.detach().clone())
        assert torch.allclose(grads[0], grads[1], atol=1e-10)

    @needs_kernels
    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    def test_the_fused_reconstruction_matches_an_axis_at_a_time(
        self, name, mode, monkeypatch
    ):
        """Test the fused inverse against the per-axis path it stands in for."""
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        bands = dwtn(x, name, mode, (-2, -1))
        fused = idwtn(bands, name, mode, (-2, -1), output_size=(16, 16))
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        per_axis = idwtn(bands, name, mode, (-2, -1), output_size=(16, 16))
        assert torch.allclose(fused, per_axis, atol=1e-11)

    @needs_kernels
    @pytest.mark.parametrize("name", ["haar", "db2", "bior2.2"])
    @pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
    def test_the_fused_reconstruction_carries_a_gradient(
        self, name, mode, monkeypatch
    ):
        """Test that the fused inverse agrees with torch on gradients."""
        base = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        grads = []
        for use in (False, True):
            monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", use)
            x = base.clone().requires_grad_(True)
            out = idwtn(dwtn(x, name, mode, (-2, -1)), name, mode, (-2, -1),
                        output_size=(16, 16))
            (out**2).sum().backward()
            grads.append(x.grad.detach().clone())
        assert torch.allclose(grads[0], grads[1], atol=1e-10)

    @needs_kernels
    @pytest.mark.parametrize("name", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
    def test_the_transforms_write_into_storage_they_are_given(self, name, mode):
        """Test that both directions take a buffer instead of making one."""
        wave = wavelet(name)
        low, high, rec_lo, rec_hi = wave.filters(torch.float64, torch.device("cpu"))
        code = MODE_TO_CODE[mode]
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        pad_lo = wave.filter_len // 2 - 1 if mode == "periodization" else wave.filter_len - 2
        out_len = 16 if mode == "periodization" else (32 + 2 * pad_lo - wave.filter_len) // 2 + 1
        made = _ext._dwt_kernels.dwt_axis(x, low, high, 0, code, pad_lo, out_len)
        given = torch.empty_like(made)
        again = _ext._dwt_kernels.dwt_axis(x, low, high, 0, code, pad_lo, out_len, given)
        assert again.data_ptr() == given.data_ptr()
        assert torch.equal(made, again)

        trim = wave.filter_len // 2 - 1 if mode == "periodization" else wave.filter_len - 2
        back = _ext._dwt_kernels.idwt_axis(made, rec_lo, rec_hi, 0, code, trim, 32)
        room = torch.empty_like(back)
        back_again = _ext._dwt_kernels.idwt_axis(
            made, rec_lo, rec_hi, 0, code, trim, 32, room
        )
        assert back_again.data_ptr() == room.data_ptr()
        assert torch.equal(back, back_again)

    @needs_kernels
    def test_storage_of_the_wrong_shape_is_refused(self):
        """Test that a buffer the transform cannot fill is reported, not filled."""
        wave = wavelet("db2")
        low, high, _, _ = wave.filters(torch.float64, torch.device("cpu"))
        x = torch.randn(2, 1, 32, 32, dtype=torch.float64)
        wrong = torch.empty(2, 2, 8, 32, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="shape"):
            _ext._dwt_kernels.dwt_axis(x, low, high, 0, 0, 2, 17, wrong)

    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_the_host_reconstructs_through_the_fused_kernel(self, dimensions):
        """Test that every rank the kernel serves is taken on the host."""
        assert _ext.idwt_nd_applies(torch.device("cpu"), dimensions)

    def test_the_fused_reconstruction_is_declined_off_the_host(self):
        """Test the calls the fused inverse is not used for."""
        assert not _ext.idwt_nd_applies(torch.device("cuda"), 2)
        assert not _ext.idwt_nd_applies(torch.device("cpu"), 4)

    def test_the_host_keeps_the_transposed_convolution(self):
        """Test that the inverse kernel is not used where it loses to torch."""
        assert not _ext.idwt_kernel_applies(torch.device("cpu"))
        assert _ext.idwt_kernel_applies(torch.device("cuda"))

    def test_an_exact_dtype_falls_back(self):
        """Test that a type the kernels cannot serve is left to torch."""
        assert not _ext.kernels_available(torch.device("cpu"), torch.int32)

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_the_two_paths_agree(self, name, mode, dimensions, monkeypatch):
        """Test the analysis against the implementation it replaces."""
        shape = tuple(8 + 2 * i for i in range(dimensions))
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, 3, *shape, dtype=torch.float64)
        compiled = dwtn(x, name, mode, axes)
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        reference = dwtn(x, name, mode, axes)
        for key in subband_keys(dimensions):
            assert torch.allclose(compiled[key], reference[key], atol=1e-11)

    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("odd", [False, True])
    def test_the_round_trip_is_exact(self, name, mode, odd):
        """Test that the compiled analysis still inverts exactly."""
        shape = (12 + odd, 14 + odd)
        x = torch.randn(2, 3, *shape, dtype=torch.float64)
        bands = dwtn(x, name, mode, (-2, -1))
        assert torch.allclose(
            idwtn(bands, name, mode, (-2, -1), shape), x, atol=1e-9
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_the_adjoint_identity_holds(self, mode):
        """Test that the hand-written backward really is the adjoint."""
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64, requires_grad=True)
        bands = dwtn(x, "db2", mode, (-2, -1))
        cotangents = {k: torch.randn_like(v) for k, v in bands.items()}
        lhs = sum(float((bands[k] * cotangents[k]).sum().detach()) for k in bands)
        torch.autograd.backward(list(bands.values()), list(cotangents.values()))
        assert lhs == pytest.approx(float((x.detach() * x.grad).sum()), abs=1e-10)

    @pytest.mark.parametrize("mode", MODES)
    def test_the_gradients_match_the_torch_path(self, mode, monkeypatch):
        """Test the backward against the one autograd derives for itself."""
        x = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)

        def objective(tensor):
            """A scalar whose cotangent varies with the coefficients themselves."""
            bands = dwtn(tensor, "db2", mode, (-2, -1))
            return sum((bands[key] ** 2).sum() for key in subband_keys(2))

        objective(x).backward()
        compiled = x.grad.clone()
        x.grad = None
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        objective(x).backward()
        assert torch.allclose(compiled, x.grad, atol=1e-11)

    @pytest.mark.parametrize("mode", MODES)
    def test_gradcheck_passes_through_the_kernel(self, mode):
        """Test the compiled gradients against finite differences."""
        x = torch.randn(1, 2, 6, 8, dtype=torch.float64, requires_grad=True)

        def run(tensor):
            return tuple(dwtn(tensor, "db2", mode, (-2, -1)).values())

        assert torch.autograd.gradcheck(run, (x,))

    @pytest.mark.parametrize("mode", MODES)
    def test_second_order_gradients_work(self, mode):
        """Test that the backward is itself differentiable."""
        x = torch.randn(1, 1, 6, 8, dtype=torch.float64, requires_grad=True)

        def run(tensor):
            return tuple(dwtn(tensor, "haar", mode, (-2, -1)).values())

        assert torch.autograd.gradgradcheck(run, (x,))

    def test_a_learnable_filter_bank_is_refused(self):
        """Test that the kernels say so rather than returning a wrong gradient."""
        from chuchichaestli.dwt.wavelet import Wavelet

        wavelet = Wavelet.from_name("db2")
        # instances are shared, so the bank is cloned before it is marked
        dec_lo, dec_hi, _, _ = (
            filt.clone() for filt in wavelet.filters(torch.float64, "cpu")
        )
        with pytest.raises(ValueError, match="filters are constants"):
            _ext.dwt_axis(
                torch.randn(1, 1, 8, dtype=torch.float64),
                dec_lo.requires_grad_(True),
                dec_hi,
                0,
                "zero",
                2,
                2,
                5,
            )

    @pytest.mark.parametrize("levels", [1, 2, 3])
    def test_multi_level_agrees(self, levels, monkeypatch):
        """Test a decomposition deep enough to chain the kernel with itself."""
        x = torch.randn(2, 2, 32, 32, dtype=torch.float64)
        compiled = wavedecn(x, "db2", "symmetric", levels, (-2, -1))
        monkeypatch.setattr(_ext, "USE_CUSTOM_KERNELS", False)
        reference = wavedecn(x, "db2", "symmetric", levels, (-2, -1))
        assert torch.allclose(compiled[0], reference[0], atol=1e-11)
        for ours, theirs in zip(compiled[1:], reference[1:], strict=True):
            for key in ours:
                assert torch.allclose(ours[key], theirs[key], atol=1e-11)


@needs_kernels
@needs_gpu
class TestDevices:
    """Tests that the compiled path agrees across devices."""

    @pytest.fixture(autouse=True)
    def _needs_gpu_kernels(self):
        """Skip when the extension was built without the GPU sources."""
        if not _ext._dwt_kernels.has_gpu():
            pytest.skip("the extension carries no GPU kernels")

    @pytest.mark.parametrize(
        "name,mode,shape",
        [
            (name, mode, shape)
            for name, mode in itertools.product(WAVELETS, MODES)
            for shape in [(2, 3, 32), (2, 3, 16, 16), (1, 2, 8, 8, 8)]
        ],
    )
    def test_the_gpu_matches_the_cpu(self, name, mode, shape):
        """Test that the two compiled paths produce the same coefficients."""
        axes = tuple(range(2 - len(shape), 0))
        x = torch.randn(*shape, dtype=torch.float64)
        on_cpu = dwtn(x, name, mode, axes)
        on_gpu = dwtn(x.cuda(), name, mode, axes)
        for key in on_cpu:
            assert torch.allclose(on_cpu[key], on_gpu[key].cpu(), atol=1e-11)

    @pytest.mark.parametrize("name,mode", itertools.product(WAVELETS, MODES))
    def test_the_gpu_reconstructs_what_it_decomposed(self, name, mode):
        """Test that the synthesis kernel inverts the analysis kernel."""
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64, device="cuda")
        bands = dwtn(x, name, mode, (-2, -1))
        assert torch.allclose(idwtn(bands, name, mode, (-2, -1), (16, 16)), x, atol=1e-10)

    @pytest.mark.parametrize("name,levels", itertools.product(WAVELETS, [1, 3]))
    def test_the_gpu_runs_the_level_recursion(self, name, levels):
        """Test that the fused recursion agrees with the one on the host."""
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        on_cpu = wavedecn(x, name, "symmetric", levels, (-2, -1))
        on_gpu = wavedecn(x.cuda(), name, "symmetric", levels, (-2, -1))
        assert torch.allclose(on_cpu[0], on_gpu[0].cpu(), atol=1e-11)
        for ours, theirs in zip(on_cpu[1:], on_gpu[1:], strict=True):
            for key in ours:
                assert torch.allclose(ours[key], theirs[key].cpu(), atol=1e-11)

    @pytest.mark.parametrize("name,mode", itertools.product(["haar", "db4"], MODES))
    def test_the_gpu_gradient_matches_the_cpu(self, name, mode):
        """Test that the adjoint agrees with the one the host computes."""
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        grads = []
        for device in ("cpu", "cuda"):
            probe = x.clone().to(device).requires_grad_(True)
            bands = dwtn(probe, name, mode, (-2, -1))
            # squared so the cotangent varies with the coefficients themselves
            sum((bands[key] ** 2).sum() for key in subband_keys(2)).backward()
            grads.append(probe.grad.cpu())
        assert torch.allclose(grads[0], grads[1], atol=1e-11)


@needs_kernels
class TestLifting:
    """Tests for the lifting kernel, which nothing dispatches to yet."""

    @staticmethod
    def _lifted(x, axis, on_detail, coeffs, lows):
        """Split one axis by lifting, in torch, as the kernel defines it."""
        moved = x.movedim(2 + axis, -1)
        half = moved.shape[-1] // 2
        approx, detail = moved[..., 0::2], moved[..., 1::2]
        for side, filt, low in zip(on_detail, coeffs, lows, strict=True):
            target, source = (detail, approx) if side else (approx, detail)
            acc = torch.zeros_like(target)
            for i, c in enumerate(filt):
                idx = (torch.arange(half) + low + i) % half
                acc = acc + c * source.index_select(-1, idx)
            if side:
                detail = target + acc
            else:
                approx = target + acc
        stacked = torch.stack([approx, detail], dim=2).flatten(1, 2)
        return stacked.movedim(-1, 2 + axis)

    @pytest.mark.parametrize("length", [2, 4, 6, 8])
    @pytest.mark.parametrize("taps", [1, 2, 3, 4, 6])
    @pytest.mark.parametrize("low", [-3, -2, -1, 0, 1])
    @pytest.mark.parametrize("side", [0, 1])
    def test_a_step_wider_than_the_axis_is_applied_once(
        self, length, taps, low, side
    ):
        """Test the taps that wrap, where the boundary runs can overlap."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, length, dtype=torch.float64)
        filt = torch.randn(taps, dtype=torch.float64).tolist()
        got = _ext._dwt_kernels.dwt_lift_axis(
            x.contiguous(), 0, [side], [filt], [low], 1.0, 0, 1.0, 0
        )
        want = self._lifted(x, 0, [side], [filt], [low])
        assert torch.allclose(got, want, atol=1e-12)

    @pytest.mark.parametrize("name", ["haar", "db2", "db4", "db8", "coif1"])
    @pytest.mark.parametrize("shape", [(2, 1, 64), (2, 3, 32, 32)])
    def test_lifting_matches_the_convolution(self, name, shape):
        """Test that both routes to the transform agree."""
        from chuchichaestli.dwt.functional import _decompose
        from chuchichaestli.utils.arithmetic.lifting import factor

        wave = wavelet(name)
        lift = factor(wave.dec_lo, wave.dec_hi)
        low, high, _, _ = wave.filters(torch.float64, torch.device("cpu"))
        x = torch.randn(*shape, dtype=torch.float64)
        for axis in range(len(shape) - 2):
            got = _ext._dwt_kernels.dwt_lift_axis(
                x,
                axis,
                [1 if s.on_detail else 0 for s in lift.steps],
                [list(s.q.c) for s in lift.steps],
                [s.q.low for s in lift.steps],
                lift.approx[0],
                lift.approx[1],
                lift.detail[0],
                lift.detail[1],
            )
            want = _decompose(x, low, high, axis, "periodization")
            assert torch.allclose(got, want, atol=1e-9)


@needs_kernels
class TestLiftingEntryPoint:
    """The supported way in to the lifting kernel."""

    @pytest.mark.parametrize("name", ["haar", "db2", "sym4", "bior2.2"])
    @pytest.mark.parametrize("groups", [1, 3])
    def test_the_wrapper_factors_and_lays_out_as_the_kernel_does(
        self, name, groups
    ):
        """Test that it agrees with the transform it stands in for."""
        torch.manual_seed(0)
        x = torch.randn(2, groups, 32, dtype=torch.float64)
        got = _ext.lift_axis(x, name, 0)
        bands = dwtn(x, name, "periodization", (2,))
        want = torch.empty_like(got)
        want[:, 0::2] = bands["a"]
        want[:, 1::2] = bands["d"]
        assert torch.allclose(got, want, atol=1e-10)

    def test_a_wavelet_is_factored_once(self):
        """Test that the factorization is held rather than redone."""
        from chuchichaestli.dwt._ext import _lifting_arguments

        _lifting_arguments.cache_clear()
        x = torch.randn(1, 1, 16, dtype=torch.float64)
        for _ in range(4):
            _ext.lift_axis(x, "db4", 0)
        assert _lifting_arguments.cache_info().misses == 1

    def test_lifting_declines_an_odd_axis(self):
        """Test that an axis it cannot wrap is refused rather than mangled."""
        x = torch.randn(1, 1, 15)
        assert not _ext.lift_axis_applies(x.device, x.dtype, 15)
        with pytest.raises(RuntimeError, match="even host axis"):
            _ext.lift_axis(x, "db2", 0)


@needs_kernels
@needs_gpu
class TestFusedOnGpu:
    """The fused transforms, on whichever device they are handed."""

    @pytest.mark.parametrize("name", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_the_fused_reconstruction_agrees_across_devices(self, name, dimensions):
        """Test that the fused inverse gives one answer on either device."""
        torch.manual_seed(0)
        wave = wavelet(name)
        trim = wave.filter_len - 2
        shape = (2, 1) + (16,) * dimensions
        axes = tuple(range(2, 2 + dimensions))
        x = torch.randn(*shape)
        bands = dwtn(x, name, "zero", axes)
        stacked = torch.cat(
            [bands[key] for key in subband_keys(dimensions)], dim=1
        )
        _, _, rec_lo, rec_hi = wave.filters(x.dtype, x.device)
        lengths = list(shape[2:])
        host = _ext.idwt_nd(
            stacked, rec_lo, rec_hi, "zero", (trim,) * dimensions, tuple(lengths)
        )
        device = _ext.idwt_nd(
            stacked.cuda(), rec_lo.cuda(), rec_hi.cuda(), "zero",
            (trim,) * dimensions, tuple(lengths),
        )
        assert torch.allclose(host, device.cpu(), atol=1e-5)
        assert torch.allclose(host, x, atol=1e-4)


@needs_kernels
class TestLowpassOnly:
    """The decomposition that skips the detail bands it would discard."""

    @pytest.mark.parametrize("name", ["haar", "db2", "db4", "sym4", "coif1"])
    @pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
    @pytest.mark.parametrize("length", [16, 17])
    def test_it_matches_the_low_pass_half_of_the_full_transform(
        self, name, mode, length
    ):
        """Test that dropping the detail changes nothing else."""
        if mode == "periodization" and length % 2:
            pytest.skip("the critically sampled mode needs an even axis")
        wave = wavelet(name)
        filter_len = wave.filter_len
        dec_lo, dec_hi, _, _ = wave.filters(torch.float64, torch.device("cpu"))
        torch.manual_seed(0)
        x = torch.randn(2, 3, length, dtype=torch.float64)
        if mode == "periodization":
            pad_lo, pad_hi = filter_len // 2 - 1, filter_len // 2
        else:
            pad_lo, pad_hi = filter_len - 2, filter_len - 2 + (length % 2)
        out_length = (length + pad_lo + pad_hi - filter_len) // 2 + 1
        code = MODE_TO_CODE[mode]
        full = _ext._dwt_kernels.dwt_axis(
            x, dec_lo, dec_hi, 0, code, pad_lo, out_length, None
        )
        low = _ext._dwt_kernels.dwt_lowpass_axis(
            x, dec_lo, 0, code, pad_lo, out_length, None
        )
        assert low.shape == full[:, 0::2].shape
        assert torch.allclose(low, full[:, 0::2], atol=1e-12)

    @pytest.mark.parametrize("name", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    @pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
    def test_the_approximation_agrees_with_the_full_transform(
        self, name, dimensions, mode
    ):
        """Test that the pyramid's band is the one the full transform gives."""
        from chuchichaestli.dwt.functional import dwtn_approx

        torch.manual_seed(0)
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, 3, *([16] * dimensions), dtype=torch.float64)
        got = dwtn_approx(x, name, mode, axes)
        want = dwtn(x, name, mode, axes)["a" * dimensions]
        assert torch.allclose(got, want, atol=1e-12)

    def test_the_approximation_still_carries_a_gradient(self):
        """Test that the differentiable route is taken when one is needed."""
        from chuchichaestli.dwt.functional import dwtn_approx

        x = torch.randn(1, 2, 16, 16, dtype=torch.float64, requires_grad=True)
        dwtn_approx(x, "db2", "zero", (2, 3)).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @needs_gpu
    @pytest.mark.parametrize("name", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_the_two_devices_give_one_answer(self, name, dimensions):
        """Test that the accelerator carries the same transform as the host."""
        from chuchichaestli.dwt.functional import dwtn_approx

        torch.manual_seed(0)
        axes = tuple(range(-dimensions, 0))
        x = torch.randn(2, 3, *([16] * dimensions))
        host = dwtn_approx(x, name, "zero", axes)
        device = dwtn_approx(x.cuda(), name, "zero", axes)
        assert host.shape == device.shape
        assert torch.allclose(host, device.cpu(), atol=1e-5)
        assert torch.allclose(
            device.cpu(), dwtn(x, name, "zero", axes)["a" * dimensions], atol=1e-5
        )


def _every_kernel(device, dtype=torch.float32):
    """Every kernel that writes one tensor, as `(name, call)` pairs.

    A kernel added without somewhere to write goes on this list and fails the
    tests below, rather than quietly keeping its own allocation.

    Args:
        device: Device to build the calls for.
        dtype: Type to build the calls for.
    """
    from chuchichaestli.dwt.functional import dwtn, subband_keys

    kernels = _ext._dwt_kernels
    wave = wavelet("db2")
    length, dims = 16, 2
    trim = wave.filter_len - 2
    out_length = (length + 2 * trim - wave.filter_len) // 2 + 1
    dev = torch.device(device)
    dec_lo, dec_hi, rec_lo, rec_hi = wave.filters(dtype, dev)
    torch.manual_seed(0)
    x = torch.randn(2, 3, *([length] * dims), dtype=dtype, device=dev)
    coeffs = kernels.dwt_nd(
        x, dec_lo, dec_hi, 0, [trim] * dims, [out_length] * dims, None
    )
    bands = dwtn(x, "db2", "zero", (2, 3))
    stacked = torch.cat([bands[key] for key in subband_keys(dims)], dim=1)
    lane = torch.randn(2, 3, length, dtype=dtype, device=dev)
    pairs = [
        ("dwt_axis", lambda o: kernels.dwt_axis(
            x, dec_lo, dec_hi, 0, 0, trim, out_length, o)),
        ("dwt_lowpass_axis", lambda o: kernels.dwt_lowpass_axis(
            x, dec_lo, 0, 0, trim, out_length, o)),
        ("dwt_nd", lambda o: kernels.dwt_nd(
            x, dec_lo, dec_hi, 0, [trim] * dims, [out_length] * dims, o)),
        ("haar_nd", lambda o: kernels.haar_nd(x, 2**-0.5, o)),
        ("idwt_axis", lambda o: kernels.idwt_axis(
            coeffs, rec_lo, rec_hi, 0, 0, trim, length, o)),
        ("idwt_nd", lambda o: kernels.idwt_nd(
            stacked, rec_lo, rec_hi, 0, [trim] * dims, [length] * dims, o)),
    ]
    if device == "cpu":
        pairs.append(("dwt_lift_axis", lambda o: kernels.dwt_lift_axis(
            lane, 0, [0], [[0.5]], [0], 1.0, 0, 1.0, 0, o)))
    return pairs


@needs_kernels
class TestCallerOwnedStorage:
    """Every kernel takes storage from the caller rather than making its own."""

    @staticmethod
    def _devices():
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_every_kernel_writes_into_the_buffer_it_is_given(self):
        """Test that the result lands in the caller's tensor, unchanged."""
        for device in self._devices():
            for name, call in _every_kernel(device):
                want = call(None)
                buffer = torch.empty_like(want)
                got = call(buffer)
                assert got.data_ptr() == buffer.data_ptr(), name
                assert torch.allclose(want, got, atol=1e-6), name

    def test_a_buffer_of_the_wrong_shape_is_refused(self):
        """Test that a mis-shaped buffer raises rather than being resized."""
        for device in self._devices():
            for name, call in _every_kernel(device):
                want = call(None)
                wrong = torch.empty(
                    tuple(s + 1 for s in want.shape),
                    dtype=want.dtype,
                    device=want.device,
                )
                with pytest.raises(RuntimeError):
                    call(wrong)

    def test_a_buffer_of_the_wrong_type_is_refused(self):
        """Test that a mis-typed buffer raises rather than being converted."""
        for device in self._devices():
            for name, call in _every_kernel(device):
                want = call(None)
                other = (
                    torch.float64 if want.dtype == torch.float32
                    else torch.float32
                )
                with pytest.raises(RuntimeError):
                    call(torch.empty_like(want, dtype=other))

    @needs_gpu
    def test_a_buffer_on_the_other_device_is_refused(self):
        """Test that storage the kernel cannot reach raises."""
        for name, call in _every_kernel("cuda"):
            want = call(None)
            with pytest.raises(RuntimeError, match="device"):
                call(torch.empty(want.shape, dtype=want.dtype, device="cpu"))


@needs_kernels
class TestGradientsKeepTheKernel:
    """The compiled path is not abandoned as soon as a gradient is wanted."""

    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_the_approximation_still_calls_the_kernel_under_a_gradient(
        self, dimensions
    ):
        """Test that recording a gradient does not fall back to convolution."""
        from chuchichaestli.dwt.functional import dwtn_approx

        kernels = _ext._dwt_kernels
        real = kernels.dwt_lowpass_axis
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        kernels.dwt_lowpass_axis = counted
        try:
            x = torch.randn(
                1, 2, *([16] * dimensions), dtype=torch.float64,
                requires_grad=True,
            )
            dwtn_approx(
                x, "db2", "zero", tuple(range(-dimensions, 0))
            ).sum().backward()
        finally:
            kernels.dwt_lowpass_axis = real
        assert len(calls) == dimensions
        assert x.grad is not None

    @pytest.mark.parametrize("name", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
    def test_the_approximation_gradient_matches_the_convolution(self, name, mode):
        """Test the adjoint against the route that never takes a kernel."""
        from chuchichaestli.dwt.functional import dwtn_approx

        torch.manual_seed(0)
        axes = (2, 3)
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64, requires_grad=True)
        plain = x.detach().clone().requires_grad_(True)
        band = dwtn_approx(x, name, mode, axes)
        seed = torch.randn_like(band)
        band.backward(seed)
        with_torch = _ext.USE_CUSTOM_KERNELS
        _ext.USE_CUSTOM_KERNELS = False
        try:
            dwtn_approx(plain, name, mode, axes).backward(seed)
        finally:
            _ext.USE_CUSTOM_KERNELS = with_torch
        assert torch.allclose(x.grad, plain.grad, atol=1e-10)

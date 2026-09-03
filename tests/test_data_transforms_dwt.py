# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the wavelet data transforms."""

import pickle

import pytest
import torch

from chuchichaestli.data.transforms.wavelet import (
    InvWaveletDecompose,
    InvWaveletPacket,
    WaveletDecompose,
    WaveletPacket,
)


NDIMS = [1, 2, 3]
WAVELETS = ["haar", "db2", "bior2.2"]
MODES = ["zero", "symmetric", "periodization", "antireflect"]


class TestWaveletDecompose:
    """Tests for the single-level transform."""

    @pytest.mark.parametrize("ndim", NDIMS)
    def test_the_subbands_land_on_one_new_axis(self, ndim):
        """Test the shape of the decomposition."""
        transform = WaveletDecompose("haar", ndim=ndim, mode="periodization")
        x = torch.randn(3, *([16] * ndim))
        out = transform.transform(x)
        assert out.shape == (3, 2**ndim, *([8] * ndim))

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("name", WAVELETS)
    @pytest.mark.parametrize("mode", MODES)
    def test_it_round_trips(self, ndim, name, mode):
        """Test that reverting a decomposition returns the input."""
        transform = WaveletDecompose(name, ndim=ndim, mode=mode)
        x = torch.randn(2, 3, *([12] * ndim), dtype=torch.float64)
        assert torch.allclose(transform.revert(transform.transform(x)), x, atol=1e-9)

    @pytest.mark.parametrize("ndim", NDIMS)
    def test_it_round_trips_odd_axes(self, ndim):
        """Test the lengths a decomposition cannot recover on its own."""
        transform = WaveletDecompose("db2", ndim=ndim, mode="symmetric")
        shape = tuple(9 + i for i in range(ndim))
        x = torch.randn(2, *shape, dtype=torch.float64)
        assert torch.allclose(transform.revert(transform.transform(x)), x, atol=1e-9)

    def test_leading_axes_pass_through(self):
        """Test that only the trailing spatial axes are touched."""
        transform = WaveletDecompose("haar", ndim=2, mode="periodization")
        out = transform.transform(torch.randn(5, 4, 3, 16, 16))
        assert out.shape == (5, 4, 3, 4, 8, 8)

    def test_the_subband_axis_can_be_placed(self):
        """Test that the subband axis goes where it is asked to."""
        transform = WaveletDecompose("haar", ndim=2, mode="periodization", stack_dim=0)
        assert transform.transform(torch.randn(3, 16, 16)).shape == (4, 3, 8, 8)

    def test_it_works_as_a_plain_transform(self):
        """Test the `__call__` path every v2 transform offers."""
        transform = WaveletDecompose("haar", ndim=2, mode="periodization")
        assert transform(torch.randn(3, 16, 16)).shape == (3, 4, 8, 8)

    def test_a_single_level_is_taken(self):
        """Test that the transform is the one whose subbands share a resolution."""
        assert WaveletDecompose("haar").levels == 1

    @pytest.mark.parametrize("ndim", [0, 4])
    def test_an_unsupported_rank_raises(self, ndim):
        """Test the limit the convolutions impose."""
        with pytest.raises(ValueError, match="one to three axes"):
            WaveletDecompose("haar", ndim=ndim)


class TestWaveletPacket:
    """Tests for the deeper decomposition that keeps one resolution."""

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("levels", [1, 2, 3])
    def test_every_subband_is_split_again(self, ndim, levels):
        """Test that a packet decomposition splits all bands, not only the coarsest."""
        transform = WaveletPacket("haar", ndim=ndim, mode="periodization", levels=levels)
        x = torch.randn(2, *([16] * ndim))
        out = transform.transform(x)
        assert out.shape == (2, 2 ** (ndim * levels), *([16 // 2**levels] * ndim))

    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("mode", MODES)
    def test_it_round_trips(self, ndim, mode):
        """Test that reverting a packet decomposition returns the input."""
        transform = WaveletPacket("db2", ndim=ndim, mode=mode, levels=2)
        x = torch.randn(2, *([16] * ndim), dtype=torch.float64)
        assert torch.allclose(transform.revert(transform.transform(x)), x, atol=1e-9)

    def test_one_level_matches_the_single_level_transform(self):
        """Test that a one-level packet is a plain decomposition."""
        x = torch.randn(2, 16, 16, dtype=torch.float64)
        packet = WaveletPacket("db2", ndim=2, levels=1).transform(x)
        plain = WaveletDecompose("db2", ndim=2).transform(x)
        assert torch.allclose(packet, plain)

    def test_a_non_positive_level_count_raises(self):
        """Test that a decomposition needs at least one level."""
        with pytest.raises(ValueError, match="at least one level"):
            WaveletPacket("haar", levels=0)


class TestInverses:
    """Tests for the transforms that run the decomposition backwards."""

    @pytest.mark.parametrize(
        "forward_cls,inverse_cls",
        [(WaveletDecompose, InvWaveletDecompose), (WaveletPacket, InvWaveletPacket)],
    )
    def test_get_inverse_returns_the_paired_transform(self, forward_cls, inverse_cls):
        """Test the type the inverse helper hands back, in both directions."""
        forward = forward_cls("db2", ndim=2)
        inverse = forward.get_inverse()
        assert isinstance(inverse, inverse_cls)
        assert isinstance(inverse.get_inverse(), forward_cls)

    @pytest.mark.parametrize("ndim", NDIMS)
    def test_the_inverse_composes_to_the_identity(self, ndim):
        """Test that a transform and its inverse cancel."""
        forward = WaveletDecompose("db2", ndim=ndim, mode="symmetric")
        x = torch.randn(2, *([12] * ndim), dtype=torch.float64)
        coefficients = forward.transform(x)
        inverse = forward.get_inverse()
        assert torch.allclose(inverse.transform(coefficients), x, atol=1e-9)
        assert torch.allclose(inverse.revert(x), coefficients, atol=1e-9)

    @pytest.mark.parametrize("ndim", NDIMS)
    def test_the_packet_inverse_composes_to_the_identity(self, ndim):
        """Test the inverse of the deeper decomposition."""
        forward = WaveletPacket("db2", ndim=ndim, mode="symmetric", levels=2)
        x = torch.randn(2, *([16] * ndim), dtype=torch.float64)
        coefficients = forward.transform(x)
        inverse = forward.get_inverse()
        assert inverse.levels == 2
        assert torch.allclose(inverse.transform(coefficients), x, atol=1e-9)
        assert torch.allclose(inverse.revert(x), coefficients, atol=1e-9)

    @pytest.mark.parametrize(
        "forward_cls", [WaveletDecompose, WaveletPacket]
    )
    def test_the_inverse_reports_rather_than_records_the_shape(self, forward_cls):
        """Test that an inverse reads the shape it was handed, never overwriting it."""
        forward = forward_cls("haar", ndim=2, mode="periodization")
        forward.transform(torch.randn(2, 16, 16))
        inverse = forward.get_inverse()
        assert inverse.make_params([torch.randn(2, 4, 8, 8)]) == {"sizes": (16, 16)}
        assert inverse.sizes == (16, 16)

    def test_the_inverse_carries_the_recorded_shape(self):
        """Test that the inverse can revert what the forward transform produced."""
        forward = WaveletDecompose("db2", ndim=2, mode="symmetric")
        forward.transform(torch.randn(2, 12, 12, dtype=torch.float64))
        assert forward.get_inverse().sizes == (12, 12)


class TestSizeBookkeeping:
    """Tests for the spatial shape a reconstruction needs."""

    def test_reverting_before_transforming_raises(self):
        """Test that the shape has to come from somewhere."""
        transform = WaveletDecompose("haar", ndim=2)
        with pytest.raises(ValueError, match="shape to revert to is unknown"):
            transform.revert(torch.randn(2, 4, 8, 8))

    def test_a_declared_shape_is_enough(self):
        """Test that the constructor argument stands in for a recorded shape."""
        transform = WaveletDecompose("haar", ndim=2, mode="periodization", sizes=(16, 16))
        assert transform.revert(torch.randn(2, 4, 8, 8)).shape == (2, 16, 16)

    def test_a_declared_shape_that_does_not_fit_raises(self):
        """Test that a declared shape is a promise the input has to keep."""
        transform = WaveletDecompose("haar", ndim=2, sizes=(16, 16))
        with pytest.raises(ValueError, match="cannot both be reverted"):
            transform.transform(torch.randn(2, 8, 8))

    def test_a_changed_shape_warns(self):
        """Test that reusing one instance across shapes is reported."""
        transform = WaveletDecompose("haar", ndim=2)
        transform.transform(torch.randn(2, 16, 16))
        with pytest.warns(UserWarning, match="previously ran on"):
            transform.transform(torch.randn(2, 8, 8))

    def test_make_params_records_the_shape(self):
        """Test the hook a collate calls once per batch."""
        transform = WaveletDecompose("haar", ndim=2)
        params = transform.make_params([torch.randn(2, 16, 16)])
        assert params == {"sizes": (16, 16)}
        assert transform.sizes == (16, 16)

    def test_a_wrong_subband_count_raises(self):
        """Test that the reconstruction checks what it was handed."""
        transform = WaveletDecompose("haar", ndim=2, sizes=(16, 16))
        with pytest.raises(ValueError, match="stacks 4 subbands"):
            transform.revert(torch.randn(2, 3, 8, 8))


class TestStateAndIntegration:
    """Tests for pickling and for the collates the transforms feed."""

    def test_pickling_drops_the_cached_wavelet(self):
        """Test that a worker rebuilds the filters on its own device."""
        transform = WaveletDecompose("db2", ndim=2, mode="periodization")
        x = torch.randn(2, 16, 16, dtype=torch.float64)
        expected = transform.transform(x)
        clone = pickle.loads(pickle.dumps(transform))
        assert clone._coerced is None
        assert torch.allclose(clone.transform(x), expected)

    def test_the_output_is_a_tensor(self):
        """Test the property that lets the result collate like any other field.

        A structure of subbands would survive the collate's walk but break the
        next transform in a chain, which looks for a tensor to size itself by.
        """
        out = WaveletDecompose("haar", ndim=2).transform(torch.randn(2, 16, 16))
        assert isinstance(out, torch.Tensor)

    def test_it_runs_inside_a_sequence_collate(self):
        """Test that paired fields are decomposed identically."""
        from chuchichaestli.data.collate import SequenceCollate

        transform = WaveletDecompose("haar", ndim=2, mode="periodization")
        samples = [
            {"image": torch.arange(64.0).reshape(1, 8, 8), "label": torch.arange(64.0).reshape(1, 8, 8)}
            for _ in range(2)
        ]
        batch = SequenceCollate(transform=transform)(samples)
        assert batch["image"].shape == (2, 1, 4, 4, 4)
        assert torch.equal(batch["image"], batch["label"])

    def test_it_chains_after_another_transform(self):
        """Test that a decomposition does not starve the chain it sits in.

        `SequentialTransform` threads each child's output into the next child's
        `make_params`, so a transform that returned a structure rather than a
        tensor would break whatever follows it.
        """
        from chuchichaestli.data.collate import SequenceCollate
        from chuchichaestli.data.transforms import RandomCropND, SequentialTransform

        chain = SequentialTransform(
            RandomCropND((8, 8)), WaveletDecompose("haar", ndim=2, mode="periodization")
        )
        batch = SequenceCollate(transform=chain)([{"x": torch.randn(3, 16, 16)}])
        assert batch["x"].shape == (1, 3, 4, 4, 4)

    def test_it_chains_before_another_transform(self):
        """Test that a transform after a decomposition still sizes itself."""
        from chuchichaestli.data.collate import SequenceCollate
        from chuchichaestli.data.transforms import Affine, SequentialTransform

        chain = SequentialTransform(
            WaveletDecompose("haar", ndim=2, mode="periodization"), Affine(2.0, 0.0)
        )
        batch = SequenceCollate(transform=chain)([{"x": torch.randn(3, 16, 16)}])
        assert batch["x"].shape == (1, 3, 4, 8, 8)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_dtype_is_preserved(self, dtype):
        """Test that the transform keeps the precision it was given."""
        out = WaveletDecompose("db2", ndim=2).transform(torch.randn(2, 8, 8, dtype=dtype))
        assert out.dtype is dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_it_runs_on_the_input_device(self):
        """Test that the filters follow the input onto the accelerator."""
        transform = WaveletDecompose("db2", ndim=2)
        out = transform.transform(torch.randn(2, 8, 8, device="cuda"))
        assert out.device.type == "cuda"

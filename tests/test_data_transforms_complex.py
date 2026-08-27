# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the complex/real view transforms."""

import pickle
import pytest
import torch
from chuchichaestli.data.transforms.complex import ComplexCollapse, ComplexExpand


class TestComplexExpand:
    """Tests for the forward expansion."""

    def test_appends_a_trailing_axis_of_two(self):
        """The complex axis becomes a trailing real/imaginary axis."""
        z = torch.randn(3, 4, dtype=torch.complex64)
        assert ComplexExpand().transform(z).shape == (3, 4, 2)

    def test_real_dtype_follows_the_complex_one(self):
        """complex64 expands to float32, complex128 to float64."""
        expand = ComplexExpand()
        assert expand.transform(torch.zeros(2, dtype=torch.complex64)).dtype is (
            torch.float32
        )
        assert expand.transform(torch.zeros(2, dtype=torch.complex128)).dtype is (
            torch.float64
        )

    def test_components_are_real_and_imaginary(self):
        """The two slots hold the real and imaginary parts, in that order."""
        z = torch.randn(5, dtype=torch.complex64)
        out = ComplexExpand().transform(z)
        assert torch.equal(out[..., 0], z.real)
        assert torch.equal(out[..., 1], z.imag)

    def test_round_trips(self):
        """Expanding then reverting recovers the input exactly."""
        z = torch.randn(2, 3, 4, dtype=torch.complex64)
        tf = ComplexExpand()
        assert torch.equal(tf.revert(tf.transform(z)), z)

    def test_expansion_is_a_view(self):
        """The default layout shares storage with the input."""
        z = torch.randn(4, dtype=torch.complex64)
        assert ComplexExpand().transform(z).data_ptr() == z.data_ptr()

    def test_callable_as_a_plain_transform(self):
        """Calling the instance expands, so it works as a collate transform."""
        assert ComplexExpand()(torch.randn(3, dtype=torch.complex64)).shape == (3, 2)


class TestDim:
    """Tests for placing the real/imaginary axis away from the end."""

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
    def test_axis_lands_where_asked(self, dim):
        """The size-2 axis appears at `dim` of the expanded tensor."""
        z = torch.randn(3, 4, dtype=torch.complex64)
        out = ComplexExpand(dim=dim).transform(z)
        assert out.ndim == 3
        assert out.shape[dim] == 2

    @pytest.mark.parametrize("dim", [0, 1, -1, -2])
    def test_round_trips_for_any_dim(self, dim):
        """`revert` undoes the move as well as the view."""
        z = torch.randn(2, 3, dtype=torch.complex64)
        tf = ComplexExpand(dim=dim)
        assert torch.equal(tf.revert(tf.transform(z)), z)

    def test_channel_first_matches_a_manual_stack(self):
        """`dim=0` is the real/imaginary-as-channels layout."""
        z = torch.randn(3, 4, dtype=torch.complex64)
        assert torch.equal(
            ComplexExpand(dim=0).transform(z), torch.stack((z.real, z.imag))
        )

    def test_non_contiguous_input_still_collapses(self):
        """A strided input takes the copying fallback rather than failing."""
        real = torch.stack((torch.randn(4, 5), torch.randn(4, 5)))
        out = ComplexExpand(dim=0).revert(real)
        assert out.shape == (4, 5)
        assert torch.equal(out.real, real[0])
        assert torch.equal(out.imag, real[1])


class TestComplexCollapse:
    """Tests for the inverse transform."""

    def test_transform_collapses(self):
        """The inverse's forward pass produces a complex tensor."""
        out = ComplexCollapse().transform(torch.randn(3, 4, 2))
        assert out.shape == (3, 4)
        assert torch.is_complex(out)

    def test_revert_expands(self):
        """The inverse's revert mirrors the forward class's transform."""
        z = torch.randn(3, dtype=torch.complex64)
        assert ComplexCollapse().revert(z).shape == (3, 2)

    def test_round_trips(self):
        """Collapsing then reverting recovers the input exactly."""
        x = torch.randn(2, 3, 2)
        tf = ComplexCollapse()
        assert torch.equal(tf.revert(tf.transform(x)), x)

    def test_inverts_the_forward_class(self):
        """The pair composes to the identity."""
        z = torch.randn(4, 5, dtype=torch.complex128)
        assert torch.equal(ComplexCollapse().transform(ComplexExpand().transform(z)), z)


class TestStrictness:
    """Tests for input validation and the passthrough escape hatch."""

    def test_real_input_to_expand_raises(self):
        """Expanding a real tensor is a mistake by default."""
        with pytest.raises(TypeError, match="expected a complex tensor"):
            ComplexExpand().transform(torch.randn(3))

    def test_complex_input_to_collapse_raises(self):
        """Collapsing an already-complex tensor is a mistake by default."""
        with pytest.raises(TypeError, match="expected a real tensor"):
            ComplexCollapse().transform(torch.randn(3, dtype=torch.complex64))

    def test_wrong_axis_size_raises(self):
        """The collapsed axis must be of size 2."""
        with pytest.raises(ValueError, match="to be of size 2"):
            ComplexCollapse().transform(torch.randn(3, 4))

    def test_lenient_expand_passes_real_through(self):
        """`strict=False` leaves real tensors untouched."""
        x = torch.randn(3, 4)
        assert ComplexExpand(strict=False).transform(x) is x

    def test_lenient_collapse_passes_mismatched_shape_through(self):
        """`strict=False` leaves tensors without a size-2 axis untouched."""
        x = torch.randn(3, 4)
        assert ComplexCollapse(strict=False).transform(x) is x

    def test_lenient_mode_still_converts_complex(self):
        """Leniency only widens what is skipped, not what is done."""
        z = torch.randn(3, dtype=torch.complex64)
        assert ComplexExpand(strict=False).transform(z).shape == (3, 2)

    def test_scalar_input_to_collapse_is_rejected(self):
        """A 0-d tensor has no axis to collapse."""
        with pytest.raises(ValueError, match="to be of size 2"):
            ComplexCollapse().transform(torch.tensor(1.0))

    def test_out_of_range_dim_is_rejected_not_an_index_error(self):
        """An axis beyond the input rank reports the same way as a wrong size."""
        with pytest.raises(ValueError, match="to be of size 2"):
            ComplexCollapse(dim=2).transform(torch.randn(4))

    def test_lenient_collapse_passes_a_too_small_rank_through(self):
        """`strict=False` must not trip over `dim` exceeding the input rank."""
        x = torch.randn(4)
        assert ComplexCollapse(dim=2, strict=False).transform(x) is x


class TestState:
    """Tests for picklability, needed for DataLoader workers."""

    @pytest.mark.parametrize("cls", [ComplexExpand, ComplexCollapse])
    def test_pickles(self, cls):
        """The transform survives a round trip through pickle."""
        restored = pickle.loads(pickle.dumps(cls(dim=1, strict=False)))
        assert isinstance(restored, cls)
        assert restored.dim == 1
        assert restored.strict is False


class TestDeviceAndDimValidation:
    """Regressions for out-of-range `dim` handling."""

    @pytest.mark.parametrize("dim", [9, -9])
    @pytest.mark.parametrize("strict", [True, False])
    def test_expand_rejects_out_of_range_dim(self, dim, strict):
        """An invalid `dim` is a config error, so `strict` must not mask it."""
        with pytest.raises(ValueError, match="out of range"):
            ComplexExpand(dim=dim, strict=strict).transform(
                torch.randn(4, dtype=torch.complex64)
            )

    def test_expand_accepts_the_edge_of_the_range(self):
        """`dim` may address the axis that expansion itself adds."""
        z = torch.randn(4, 3, dtype=torch.complex64)
        assert ComplexExpand(dim=2).transform(z).shape == (4, 3, 2)
        assert ComplexExpand(dim=-3).transform(z).shape == (2, 4, 3)

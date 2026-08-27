# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the basis projection transforms."""

import pickle
import pytest
import torch
from chuchichaestli.data.transforms.basis import (
    BASIS_REGISTRY,
    BasisProjection,
    InvBasisProjection,
)


def smooth_signal(n=64):
    """A smooth, well-approximable test signal on [-1, 1]."""
    t = torch.linspace(-1, 1, n, dtype=torch.float64)
    return torch.exp(-(t**2)) * torch.cos(2 * t)


class TestBasisRegistry:
    """Tests for the built-in basis families."""

    @pytest.mark.parametrize("name", sorted(BASIS_REGISTRY))
    def test_design_matrix_shape(self, name):
        """Each builder returns an (n, m) matrix."""
        assert BASIS_REGISTRY[name](16, 5).shape == (16, 5)

    @pytest.mark.parametrize("name", sorted(BASIS_REGISTRY))
    def test_complete_basis_round_trips_exactly(self, name):
        """A complete basis (m == n) reconstructs the signal exactly."""
        proj = BasisProjection({0: (name, 12)})
        x = torch.randn(12, dtype=torch.float64)
        assert torch.allclose(proj.revert(proj.transform(x)), x, atol=1e-9)

    def test_chebyshev_matches_closed_form(self):
        """T_n(x) = cos(n arccos x) on the node grid."""
        basis = BASIS_REGISTRY["chebyshev"](9, 4)
        x = torch.linspace(-1, 1, 9, dtype=torch.float64)
        for k in range(4):
            assert torch.allclose(basis[:, k], torch.cos(k * torch.arccos(x)))

    def test_legendre_matches_known_values(self):
        """P_0..P_3 match their closed forms."""
        basis = BASIS_REGISTRY["legendre"](11, 4)
        x = torch.linspace(-1, 1, 11, dtype=torch.float64)
        assert torch.allclose(basis[:, 0], torch.ones_like(x))
        assert torch.allclose(basis[:, 1], x)
        assert torch.allclose(basis[:, 2], 0.5 * (3 * x**2 - 1))
        assert torch.allclose(basis[:, 3], 0.5 * (5 * x**3 - 3 * x))


class TestBasisProjection:
    """Tests for the forward projection."""

    def test_projection_shrinks_the_axis(self):
        """The selected axis is replaced by the coefficient count."""
        proj = BasisProjection({1: ("chebyshev", 4)})
        assert proj.transform(torch.randn(3, 16, 5)).shape == (3, 4, 5)

    def test_truncation_error_decreases_with_order(self):
        """A smooth signal is better approximated at higher order."""
        x = smooth_signal()
        errors = []
        for m in (2, 4, 8, 16):
            proj = BasisProjection({0: m})
            rec = proj.revert(proj.transform(x))
            errors.append(float((x - rec).norm() / x.norm()))
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < 1e-6

    def test_multiple_axes_project_independently(self):
        """Two axes can carry different bases and orders."""
        proj = BasisProjection({1: ("chebyshev", 4), 2: ("dct", 8)})
        coeffs = proj.transform(torch.randn(2, 16, 32, dtype=torch.float64))
        assert coeffs.shape == (2, 4, 8)
        assert proj.revert(coeffs).shape == (2, 16, 32)

    def test_bare_order_uses_default_family(self):
        """An int spec resolves against the `basis` argument."""
        proj = BasisProjection({0: 3}, basis="legendre")
        assert proj.bases[0] == ("legendre", 3)

    def test_negative_axis_is_resolved(self):
        """Negative axes index from the end."""
        proj = BasisProjection({-1: ("dct", 4)})
        assert proj.transform(torch.randn(2, 3, 16)).shape == (2, 3, 4)

    def test_complex_input_round_trips(self):
        """Real bases apply to complex data component-wise."""
        proj = BasisProjection({1: ("chebyshev", 8)})
        z = torch.randn(2, 8, dtype=torch.complex128)
        assert torch.allclose(proj.revert(proj.transform(z)), z, atol=1e-10)

    def test_explicit_design_matrix(self):
        """An explicit (N, M) matrix is used verbatim."""
        basis = torch.linalg.qr(torch.randn(10, 3, dtype=torch.float64))[0]
        proj = BasisProjection({0: basis})
        coeffs = proj.transform(torch.randn(10, dtype=torch.float64))
        assert coeffs.shape == (3,)
        assert proj.lengths[0] == 10

    def test_callable_as_plain_transform(self):
        """Calling the instance projects, so it works as a collate transform."""
        assert BasisProjection({1: 4})(torch.randn(2, 8, 3)).shape == (2, 4, 3)


class TestWeightedFit:
    """Tests for the weighted least-squares path."""

    def test_zero_weight_ignores_a_corrupted_sample(self):
        """A zero-weighted outlier does not perturb the fit."""
        clean = torch.linspace(0, 1, 16, dtype=torch.float64)
        dirty = clean.clone()
        dirty[7] = 99.0
        weights = torch.ones(16, dtype=torch.float64)
        weights[7] = 0.0

        plain = BasisProjection({0: ("chebyshev", 3)})
        weighted = BasisProjection({0: ("chebyshev", 3)}, weights={0: weights})
        plain_err = (plain.revert(plain.transform(dirty)) - clean).abs().max()
        weighted_err = (weighted.revert(weighted.transform(dirty)) - clean).abs().max()
        assert weighted_err < 1e-9
        assert plain_err > 1.0

    def test_uniform_weights_match_unweighted(self):
        """Constant weights reproduce the ordinary least-squares fit."""
        x = smooth_signal(32)
        plain = BasisProjection({0: ("chebyshev", 6)})
        weighted = BasisProjection(
            {0: ("chebyshev", 6)}, weights={0: torch.ones(32, dtype=torch.float64)}
        )
        assert torch.allclose(plain.transform(x), weighted.transform(x), atol=1e-9)

    def test_wrong_weight_length_raises(self):
        """Weights must match the axis length."""
        proj = BasisProjection({0: 3}, weights={0: torch.ones(5)})
        with pytest.raises(ValueError, match="expected 16"):
            proj.transform(torch.randn(16, dtype=torch.float64))

    def test_negative_weights_raise(self):
        """Negative weights have no meaning in a least-squares fit."""
        w = torch.ones(16, dtype=torch.float64)
        w[3] = -5.0
        proj = BasisProjection({0: 3}, weights={0: w})
        with pytest.raises(ValueError, match="must be non-negative"):
            proj.transform(torch.randn(16, dtype=torch.float64))

    def test_weights_are_copied_at_construction(self):
        """Mutating the caller's tensor afterwards does not change the fit."""
        w = torch.ones(16, dtype=torch.float64)
        proj = BasisProjection({0: 3}, weights={0: w})
        w[0] = -1.0
        proj.transform(torch.randn(16, dtype=torch.float64))

    def test_weighted_matrices_are_cached(self):
        """The weighted solve is cacheable; it must not redo the SVD."""
        proj = BasisProjection({0: 6}, weights={0: torch.ones(64, dtype=torch.float64)})
        x = torch.randn(64, dtype=torch.float64)
        proj.transform(x)
        proj.transform(x)
        assert len(proj._cache) == 1

    def test_masked_weights_are_judged_on_the_whitened_matrix(self):
        """Zero weights that starve the fit must trip the conditioning check."""
        w = torch.zeros(64, dtype=torch.float64)
        w[:8] = 1.0
        proj = BasisProjection({0: ("chebyshev", 40)}, weights={0: w})
        with pytest.warns(UserWarning, match="ill-conditioned"):
            proj.transform(torch.randn(64, dtype=torch.float64))

    def test_weighted_fit_beats_the_normal_equations(self):
        """The whitened solve is accurate where B^T W B is already degraded."""
        n, m = 64, 40
        torch.manual_seed(0)
        w = torch.rand(n, dtype=torch.float64) * 0.99 + 0.01
        x = smooth_signal(n)
        proj = BasisProjection({0: ("chebyshev", m)}, weights={0: w}, cond_warn=None)
        basis = proj.design_matrix(0, n)
        root = w.sqrt().unsqueeze(1)
        reference = torch.linalg.lstsq(basis * root, root.squeeze(1) * x).solution
        got = proj.transform(x)
        assert torch.allclose(got, reference, rtol=1e-9, atol=1e-9)


class TestConditioning:
    """Tests for the ill-conditioning warning."""

    def test_warns_on_ill_conditioned_fit(self):
        """A near-singular design matrix warns."""
        with pytest.warns(UserWarning, match="ill-conditioned"):
            BasisProjection({0: ("chebyshev", 64)}).transform(
                torch.randn(64, dtype=torch.float64)
            )

    def test_no_warning_for_a_well_conditioned_fit(self):
        """A complete basis on few samples is exact and must not warn."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BasisProjection({0: ("chebyshev", 8)}).transform(
                torch.randn(8, dtype=torch.float64)
            )
        assert not caught

    def test_warning_can_be_disabled(self):
        """cond_warn=None silences the check."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BasisProjection({0: ("chebyshev", 64)}, cond_warn=None).transform(
                torch.randn(64, dtype=torch.float64)
            )
        assert not caught


class TestInvBasisProjection:
    """Tests for the inverse transform."""

    def test_transform_reconstructs(self):
        """The inverse expands coefficients back to samples."""
        proj = BasisProjection({0: ("chebyshev", 4)})
        coeffs = proj.transform(smooth_signal(32))
        inverse = InvBasisProjection({0: ("chebyshev", 4)}, lengths={0: 32})
        assert inverse.transform(coeffs).shape == (32,)

    def test_revert_projects(self):
        """The inverse's revert projects, mirroring the forward transform."""
        inverse = InvBasisProjection({0: ("chebyshev", 4)})
        assert inverse.revert(smooth_signal(32)).shape == (4,)

    def test_round_trip_matches_forward_pair(self):
        """Inverse-of-projection equals the forward class's revert."""
        x = smooth_signal(32)
        proj = BasisProjection({0: ("chebyshev", 6)})
        coeffs = proj.transform(x)
        inverse = InvBasisProjection({0: ("chebyshev", 6)}, lengths={0: 32})
        assert torch.allclose(inverse.transform(coeffs), proj.revert(coeffs))


class TestValidationAndState:
    """Tests for argument validation, length tracking and pickling."""

    def test_empty_bases_raises(self):
        """At least one axis must be given."""
        with pytest.raises(ValueError, match="at least one axis"):
            BasisProjection({})

    def test_unknown_basis_raises(self):
        """An unregistered family name is rejected."""
        with pytest.raises(ValueError, match="unknown basis"):
            BasisProjection({0: ("bogus", 3)})

    def test_non_positive_order_raises(self):
        """Orders below 1 are rejected."""
        with pytest.raises(ValueError, match="must be >= 1"):
            BasisProjection({0: 0})

    def test_order_above_axis_length_raises(self):
        """An underdetermined fit is rejected."""
        with pytest.raises(ValueError, match="underdetermined"):
            BasisProjection({0: 20}).transform(torch.randn(8, dtype=torch.float64))

    def test_non_2d_design_matrix_raises(self):
        """An explicit basis must be 2-D."""
        with pytest.raises(ValueError, match="must be 2-D"):
            BasisProjection({0: torch.randn(4)})

    def test_duplicate_axis_raises(self):
        """Two keys resolving to the same axis are rejected."""
        proj = BasisProjection({-1: 2, 1: 2})
        with pytest.raises(ValueError, match="both refer to axis"):
            proj.transform(torch.randn(3, 8, dtype=torch.float64))

    def test_weights_for_an_undeclared_axis_raise(self):
        """Weights are keyed by the `bases` key, so a stray axis is a typo."""
        with pytest.raises(ValueError, match="weights names axes"):
            BasisProjection({0: 2}, weights={1: torch.ones(8)})

    def test_weights_written_with_the_other_sign_raise(self):
        """`-1` and the equivalent positive axis are not interchangeable here."""
        with pytest.raises(ValueError, match="same sign"):
            BasisProjection({1: 2}, weights={-1: torch.ones(8)})

    def test_lengths_for_an_undeclared_axis_raise(self):
        """A length can only be declared for an axis that has a basis."""
        with pytest.raises(ValueError, match="lengths names axes"):
            BasisProjection({0: 2}, lengths={1: 8})

    def test_axis_out_of_range_raises(self):
        """An axis beyond the input rank is rejected."""
        with pytest.raises(ValueError, match="out of range"):
            BasisProjection({5: 2}).transform(torch.randn(3, 8, dtype=torch.float64))

    def test_negative_axis_out_of_range_raises(self):
        """A too-negative axis must not wrap onto an unrelated axis."""
        # -5 % 3 == 1, so wrapping would silently project the wrong dimension
        with pytest.raises(ValueError, match="out of range"):
            BasisProjection({-5: 2}).transform(torch.randn(2, 3, 4))

    def test_negative_axis_in_range_still_works(self):
        """The valid negative range keeps addressing from the end."""
        proj = BasisProjection({-2: 4})
        assert proj.transform(torch.randn(2, 16, 5)).shape == (2, 4, 5)

    def test_revert_without_length_raises(self):
        """Reverting needs the original axis length."""
        with pytest.raises(RuntimeError, match="original length"):
            BasisProjection({0: 3}).revert(torch.randn(3, dtype=torch.float64))

    def test_projection_records_length(self):
        """Projecting records the length needed to revert."""
        proj = BasisProjection({1: 4})
        proj.transform(torch.randn(2, 16, dtype=torch.float64))
        assert proj.lengths[1] == 16

    def test_reprojecting_the_same_length_is_quiet(self):
        """Re-recording an unchanged length must not warn."""
        import warnings

        proj = BasisProjection({0: 4})
        proj.transform(torch.randn(16, dtype=torch.float64))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proj.transform(torch.randn(16, dtype=torch.float64))
        assert not caught

    def test_changing_the_recorded_length_warns(self):
        """A contradicted auto-recorded length invalidates old coefficients."""
        proj = BasisProjection({0: 4})
        proj.transform(torch.randn(16, dtype=torch.float64))
        with pytest.warns(UserWarning, match="previously projected at length 16"):
            proj.transform(torch.randn(32, dtype=torch.float64))
        assert proj.lengths[0] == 32

    def test_contradicting_a_declared_length_raises(self):
        """A length passed by the caller is a promise, not an observation."""
        proj = BasisProjection({0: 4}, lengths={0: 16})
        with pytest.raises(ValueError, match="declared to have length 16"):
            proj.transform(torch.randn(32, dtype=torch.float64))

    def test_explicit_design_matrix_length_is_declared(self):
        """A design matrix pins the axis length just as `lengths` does."""
        basis = torch.linalg.qr(torch.randn(10, 3, dtype=torch.float64))[0]
        with pytest.raises(ValueError, match="declared to have length 10"):
            BasisProjection({0: basis}).transform(torch.randn(12, dtype=torch.float64))

    def test_integer_input_is_promoted(self):
        """Integer input is promoted rather than failing in the matmul."""
        coeffs = BasisProjection({0: 2}).transform(torch.arange(8))
        assert coeffs.dtype is torch.float32
        assert coeffs.shape == (2,)

    def test_conditioning_warning_names_the_axis_and_the_numbers(self):
        """The message must locate the problem on its own, without a traceback."""
        with pytest.warns(UserWarning, match="ill-conditioned") as caught:
            BasisProjection({1: ("chebyshev", 64)}).transform(
                torch.randn(3, 64, dtype=torch.float64)
            )
        message = str(caught[0].message)
        assert "axis 1" in message
        assert "64 terms on 64 samples" in message

    def test_mismatched_coefficient_count_raises(self):
        """Reverting a wrongly sized coefficient axis is rejected."""
        proj = BasisProjection({0: 4}, lengths={0: 16})
        with pytest.raises(ValueError, match="coefficients"):
            proj.revert(torch.randn(7, dtype=torch.float64))

    def test_pickles_without_the_cache(self):
        """The matrix cache is dropped but the geometry survives."""
        proj = BasisProjection({1: ("chebyshev", 4)})
        proj.transform(torch.randn(2, 8, dtype=torch.float64))
        restored = pickle.loads(pickle.dumps(proj))
        assert restored._cache == {}
        assert restored.lengths == {1: 8}
        assert restored.bases == proj.bases

    def test_cache_is_reused(self):
        """Repeated projections reuse the cached matrices."""
        proj = BasisProjection({0: 4})
        proj.transform(torch.randn(16, dtype=torch.float64))
        assert len(proj._cache) == 1
        proj.transform(torch.randn(16, dtype=torch.float64))
        assert len(proj._cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestDevicePlacement:
    """The solve runs on the design matrix's device, not the caller's."""

    def test_weighted_fit_with_non_cpu_weights(self):
        """Built-in bases are built on CPU, so GPU weights must be moved."""
        proj = BasisProjection(
            {0: ("legendre", 4)},
            weights={0: torch.rand(20, device="cuda") + 0.1},
            cond_warn=None,
        )
        out = proj.transform(torch.randn(20, device="cuda"))
        assert out.shape == (4,)
        assert out.device.type == "cuda"

    def test_weighted_result_matches_cpu(self):
        """Moving the weights must not change the fit."""
        w = torch.rand(20, dtype=torch.float64) + 0.1
        y = torch.randn(20, dtype=torch.float64)
        cpu = BasisProjection({0: ("legendre", 4)}, weights={0: w}, cond_warn=None)
        gpu = BasisProjection(
            {0: ("legendre", 4)}, weights={0: w.cuda()}, cond_warn=None
        )
        assert torch.allclose(cpu.transform(y), gpu.transform(y.cuda()).cpu())

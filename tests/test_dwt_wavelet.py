# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the wavelet filter banks."""

import pickle

import pytest
import torch

from chuchichaestli.dwt.filters import BIORTHOGONAL_BANKS, FAMILIES, ORTHOGONAL_DEC_LO
from chuchichaestli.dwt.wavelet import (
    WAVELET_REGISTRY,
    Wavelet,
    wavelet,
    wavelist,
)


# Filter banks as PyWavelets reports them, so the derivation rules are pinned to
# published values rather than to our own output.
GOLDEN_BANKS = {
    "db2": (
        (
            -0.12940952255126, 0.224143868042013, 0.836516303737808,
            0.482962913144534,
        ),
        (
            -0.482962913144534, 0.836516303737808, -0.224143868042013,
            -0.12940952255126,
        ),
        (
            0.482962913144534, 0.836516303737808, 0.224143868042013,
            -0.12940952255126,
        ),
        (
            -0.12940952255126, -0.224143868042013, 0.836516303737808,
            -0.482962913144534,
        ),
    ),
    "sym4": (
        (
            -0.075765714789273, -0.029635527645999, 0.497618667632015,
            0.803738751805916, 0.297857795605277, -0.099219543576847,
            -0.012603967262038, 0.032223100604043,
        ),
        (
            -0.032223100604043, -0.012603967262038, 0.099219543576847,
            0.297857795605277, -0.803738751805916, 0.497618667632015,
            0.029635527645999, -0.075765714789273,
        ),
        (
            0.032223100604043, -0.012603967262038, -0.099219543576847,
            0.297857795605277, 0.803738751805916, 0.497618667632015,
            -0.029635527645999, -0.075765714789273,
        ),
        (
            -0.075765714789273, 0.029635527645999, 0.497618667632015,
            -0.803738751805916, 0.297857795605277, 0.099219543576847,
            -0.012603967262038, -0.032223100604043,
        ),
    ),
    "coif1": (
        (
            -0.015655728135792, -0.072732619512526, 0.384864846864858,
            0.8525720202116, 0.337897662457482, -0.072732619512526,
        ),
        (
            0.072732619512526, 0.337897662457482, -0.8525720202116,
            0.384864846864858, 0.072732619512526, -0.015655728135792,
        ),
        (
            -0.072732619512526, 0.337897662457482, 0.8525720202116,
            0.384864846864858, -0.072732619512526, -0.015655728135792,
        ),
        (
            -0.015655728135792, 0.072732619512526, 0.384864846864858,
            -0.8525720202116, 0.337897662457482, 0.072732619512526,
        ),
    ),
    "bior2.2": (
        (
            0.0, -0.176776695296637, 0.353553390593274, 1.060660171779821,
            0.353553390593274, -0.176776695296637,
        ),
        (
            -0.0, 0.353553390593274, -0.707106781186548, 0.353553390593274, -0.0,
            0.0,
        ),
        (
            0.0, 0.353553390593274, 0.707106781186548, 0.353553390593274, 0.0, 0.0,
        ),
        (
            0.0, 0.176776695296637, 0.353553390593274, -1.060660171779821,
            0.353553390593274, 0.176776695296637,
        ),
    ),
    "rbio2.2": (
        (
            0.0, 0.0, 0.353553390593274, 0.707106781186548, 0.353553390593274, 0.0,
        ),
        (
            0.176776695296637, 0.353553390593274, -1.060660171779821,
            0.353553390593274, 0.176776695296637, 0.0,
        ),
        (
            -0.176776695296637, 0.353553390593274, 1.060660171779821,
            0.353553390593274, -0.176776695296637, 0.0,
        ),
        (
            0.0, -0.0, 0.353553390593274, -0.707106781186548, 0.353553390593274,
            -0.0,
        ),
    ),
}


class TestFilterBank:
    """Tests for the filter banks the registry builds."""

    @pytest.mark.parametrize("name", sorted(WAVELET_REGISTRY))
    def test_every_registered_wavelet_builds(self, name):
        """Test that every registered name yields a usable filter bank."""
        w = Wavelet.from_name(name)
        assert w.name == name
        assert w.filter_len > 0
        assert all(len(f) == w.filter_len for f in w.filter_bank)

    @pytest.mark.parametrize("name", sorted(WAVELET_REGISTRY))
    def test_every_registered_wavelet_reconstructs(self, name):
        """Test that every filter bank satisfies the perfect reconstruction conditions."""
        assert Wavelet.from_name(name).check_perfect_reconstruction()

    @pytest.mark.parametrize("name", sorted(GOLDEN_BANKS))
    def test_derivation_matches_published_banks(self, name):
        """Test that the quadrature mirror rules reproduce the published filters."""
        w = Wavelet.from_name(name)
        for ours, golden in zip(w.filter_bank, GOLDEN_BANKS[name], strict=True):
            assert len(ours) == len(golden)
            for a, b in zip(ours, golden, strict=True):
                assert a == pytest.approx(b, abs=1e-14)

    @pytest.mark.parametrize("name", sorted(ORTHOGONAL_DEC_LO))
    def test_orthogonal_filters_are_normalized(self, name):
        """Test the low- and high-pass sum rules of an orthogonal filter bank."""
        w = Wavelet.from_name(name)
        assert w.orthogonal
        # The published coefficients are rounded, so the sum rules hold only to
        # about the accumulated rounding error of a filter this long.
        assert sum(w.dec_lo) == pytest.approx(2**0.5, abs=1e-10)
        assert sum(w.dec_hi) == pytest.approx(0.0, abs=1e-10)
        assert sum(v * v for v in w.dec_lo) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("name", sorted(BIORTHOGONAL_BANKS))
    def test_biorthogonal_wavelets_are_not_orthogonal(self, name):
        """Test that the biorthogonal families are flagged as such."""
        w = Wavelet.from_name(name)
        assert w.biorthogonal
        assert not w.orthogonal

    @pytest.mark.parametrize("name", sorted(FAMILIES["bior"]))
    def test_reverse_biorthogonal_swaps_the_analysis_and_synthesis_sides(self, name):
        """Test that `rbioN.M` is `biorN.M` with both sides swapped and reversed."""
        bior = Wavelet.from_name(name)
        rbio = Wavelet.from_name(name.replace("bior", "rbio"))
        assert rbio.dec_lo == bior.rec_lo[::-1]
        assert rbio.rec_lo == bior.dec_lo[::-1]
        assert rbio.dec_hi == bior.rec_hi[::-1]
        assert rbio.rec_hi == bior.dec_hi[::-1]

    def test_haar_is_db1(self):
        """Test that the `haar` alias resolves to the same bank as `db1`."""
        assert Wavelet.from_name("haar").filter_bank == Wavelet.from_name("db1").filter_bank


class TestConstruction:
    """Tests for building wavelets from explicit filters."""

    def test_a_single_filter_derives_an_orthogonal_bank(self):
        """Test that passing only `dec_lo` produces an orthogonal wavelet."""
        w = Wavelet(Wavelet.from_name("db3").dec_lo)
        assert w.orthogonal
        assert w.check_perfect_reconstruction()

    def test_an_explicit_bank_is_used_as_given(self):
        """Test that four explicit filters are stored unchanged."""
        ref = Wavelet.from_name("bior2.2")
        w = Wavelet(*ref.filter_bank, name="mine", family="mine")
        assert w.filter_bank == ref.filter_bank
        assert w.name == "mine"

    def test_lengths_and_properties_agree(self):
        """Test the reported filter lengths."""
        w = Wavelet.from_name("db4")
        assert w.dec_len == w.rec_len == w.filter_len == 8

    def test_repr_names_the_wavelet(self):
        """Test that the representation carries the identifying fields."""
        text = repr(Wavelet.from_name("db2"))
        assert "db2" in text and "orthogonal" in text and "filter_len=4" in text

    def test_equality_and_hashing_use_the_bank(self):
        """Test that wavelets compare and hash by name and filter bank."""
        a, b = Wavelet.from_name("db2"), Wavelet.from_name("db2")
        assert a == b and hash(a) == hash(b)
        assert a != Wavelet.from_name("db3")
        assert a.__eq__(object()) is NotImplemented

    def test_unknown_name_raises(self):
        """Test that an unregistered name is rejected."""
        with pytest.raises(ValueError, match="Unknown wavelet"):
            Wavelet.from_name("nope")

    def test_empty_filter_raises(self):
        """Test that a wavelet needs coefficients."""
        with pytest.raises(ValueError, match="at least one filter coefficient"):
            Wavelet([])

    def test_filters_of_different_lengths_raise(self):
        """Test that the four filters must agree in length."""
        with pytest.raises(ValueError, match="same length"):
            Wavelet([1.0, 1.0], [1.0, -1.0], [1.0, 1.0, 0.0], [1.0, -1.0])

    def test_a_bank_declared_orthogonal_must_reconstruct(self):
        """Test that a bogus bank cannot claim to be orthogonal."""
        with pytest.raises(ValueError, match="does not reconstruct"):
            Wavelet([1.0, 2.0], [0.0, 1.0], [1.0, 0.0], [3.0, 1.0], orthogonal=True)


class TestRegistry:
    """Tests for the registry helpers."""

    def test_wavelist_covers_the_registry(self):
        """Test that the unfiltered listing is the whole registry."""
        assert wavelist() == sorted(WAVELET_REGISTRY)

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_wavelist_filters_by_family(self, family):
        """Test that a family listing only holds that family."""
        names = wavelist(family)
        assert names and all(Wavelet.from_name(n).family == family for n in names)

    def test_wavelist_rejects_an_unknown_family(self):
        """Test that an unknown family is reported."""
        with pytest.raises(ValueError, match="Unknown wavelet family"):
            wavelist("nope")

    def test_wavelet_coerces_names_and_passes_instances_through(self):
        """Test the coercion helper every public entry point uses."""
        w = Wavelet.from_name("db2")
        assert wavelet("db2") == w
        assert wavelet(w) is w


class TestTensorsAndState:
    """Tests for the tensor cache and pickling."""

    def test_filters_are_cached_per_dtype_and_device(self):
        """Test that repeated requests share the storage they were built into."""
        w = Wavelet.from_name("db2")
        first = w.filters(torch.float32, "cpu")
        again = w.filters(torch.float32, "cpu")
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(first, again))
        other = w.filters(torch.float64, "cpu")
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, other))

    def test_the_shared_bank_cannot_be_marked_by_a_caller(self):
        """Test that one caller marking the filters does not reach the next."""
        w = Wavelet.from_name("db2")
        w.filters(torch.float64, "cpu")[0].requires_grad_(True)
        assert not any(f.requires_grad for f in w.filters(torch.float64, "cpu"))

    def test_lookup_by_name_is_shared(self):
        """Test that a repeated lookup does not derive the bank again."""
        assert Wavelet.from_name("db4") is Wavelet.from_name("db4")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_filters_respect_the_requested_dtype(self, dtype):
        """Test that the returned tensors carry the requested dtype."""
        tensors = Wavelet.from_name("db2").filters(dtype, "cpu")
        assert all(t.dtype is dtype for t in tensors)
        assert len(tensors) == 4

    def test_pickling_drops_the_cache(self):
        """Test that a round trip through pickle rebuilds the cache lazily."""
        w = Wavelet.from_name("db2")
        w.filters(torch.float32, "cpu")
        clone = pickle.loads(pickle.dumps(w))
        assert clone == w
        assert clone._cache == {}
        assert clone.filters(torch.float32, "cpu")[0].dtype is torch.float32

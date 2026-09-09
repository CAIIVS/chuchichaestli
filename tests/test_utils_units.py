# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the unit tables and unit-aware numbers."""

import pytest
from chuchichaestli.utils.units import BYTE_UNITS, METRIC_UNITS, nbytes


class TestByteUnits:
    """The units a byte size may be given in."""

    def test_binary_units_are_powers_of_two(self):
        """A bare suffix is binary, so '1K' is 1024 bytes."""
        assert BYTE_UNITS["b"] == 1
        assert [BYTE_UNITS[u] for u in "KMGTP"] == [1 << (10 * n) for n in range(1, 6)]

    def test_suffixed_units_are_the_metric_ones(self):
        """A 'B' suffix is decimal, and shares its powers with `metric_suffix`."""
        assert BYTE_UNITS["B"] == 1
        assert all(BYTE_UNITS[f"{u}B"] == size for u, size in METRIC_UNITS.items())


class TestNbytes:
    """Unit tests for nbytes."""

    @pytest.mark.parametrize("x", ["2.0G", "2.0GB", "2.0 GB", 2147483648.0, 2147483648])
    def test_nbytes_2G(self, x):
        """Test the nbytes class."""
        b = nbytes(x)
        assert 0 < b < 10**10
        assert isinstance(b, nbytes)
        assert isinstance(b, float)
        assert b == float(b)
        assert isinstance(b.as_str(), str)
        assert isinstance(b.as_bstr(), str)
        assert isinstance(b.to("G"), nbytes)
        assert isinstance(b.to("G"), float)

    @pytest.mark.parametrize("x", [None, 0.0, 0, "GB"])
    def test_nbytes_null(self, x):
        """Test the nbytes class in edge cases."""
        b = nbytes(x)
        assert b == 0
        assert isinstance(b, nbytes)
        assert isinstance(b, float)
        assert b == float(b)
        assert isinstance(b.as_str(), str)
        assert isinstance(b.as_bstr(), str)
        assert isinstance(b.to("G"), nbytes)
        assert isinstance(b.to("G"), float)

    def test_invalid_unit_raises(self):
        """An unknown unit suffix in the string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown unit"):
            nbytes("4X")

    def test_repr(self):
        """__repr__ returns the same string as __str__ / as_str."""
        b = nbytes("1M")
        assert repr(b) == b.as_bstr()

    def test_arithmetic_preserves_type(self):
        """Class inherits float arithmetic; results are plain floats."""
        b = nbytes("1M")
        assert b + b == float(b) * 2

    def test_nbytes_size_constructor(self):
        """Passing an nbytes instance directly round-trips correctly."""
        original = nbytes("512K")
        copy = nbytes(original)
        assert copy == original
        assert isinstance(copy, nbytes)

    @pytest.mark.parametrize(
        "text,expected", [("1K", 1024), ("1KB", 1000), ("1 KB", 1000), ("1b", 1)]
    )
    def test_binary_and_decimal_units(self, text, expected):
        """A bare suffix counts in powers of two, a 'B' suffix in powers of ten."""
        assert nbytes(text) == expected

    def test_as_str_and_as_bstr_pick_their_units(self):
        """One byte count reads decimal or binary, and says which it is."""
        assert nbytes(2 * 1024**3).as_str() == "2.15GB"
        assert nbytes(2 * 1024**3).as_bstr() == "2.00G"

    def test_to_converts(self):
        """A byte count converts into any unit it can be given in."""
        assert nbytes("2G").to("M") == 2048
        assert nbytes("2GB").to("MB") == 2000

    def test_to_rejects_an_unknown_unit(self):
        """An unknown unit fails by name rather than returning nonsense."""
        with pytest.raises(ValueError, match="Unknown unit"):
            nbytes("1M").to("X")

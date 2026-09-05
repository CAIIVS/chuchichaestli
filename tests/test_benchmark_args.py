# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the shared benchmark command line."""

import argparse

import pytest
from chuchichaestli.benchmark.args import base_parser, shape, without_options


class TestShape:
    """The `x`-separated shape argument type."""

    @pytest.mark.parametrize(
        "text,expected",
        [("4096", [4096]), ("256x256", [256, 256]), ("2x3x64x64", [2, 3, 64, 64])],
    )
    def test_parses(self, text, expected):
        """A shape is read left to right, one extent per `x`-separated field."""
        assert shape(text) == expected

    @pytest.mark.parametrize("text", ["", "256x", "16x0", "-4", "256x-1", "a", "2.5", "2, 3"])
    def test_rejects(self, text):
        """Anything that is not a run of positive integers is refused by name."""
        with pytest.raises(argparse.ArgumentTypeError):
            shape(text)


class TestBaseParser:
    """The arguments every benchmark shares."""

    def test_defaults(self):
        """The defaults are the cheap, reproducible run: cpu, float32, one sweep."""
        args = base_parser(backends=["a", "b"]).parse_args([])
        assert args.device == "cpu"
        assert args.dtype == "float32"
        assert args.backends == ["a", "b"]
        assert args.repeats == 1
        assert args.threads is None
        assert (args.json, args.csv, args.plot) == (None, None, None)
        assert not args.profile and not args.perf

    def test_backends_are_checked(self):
        """A misspelled backend fails at the command line, not with a KeyError."""
        parser = base_parser(backends=["a", "b"])
        assert parser.parse_args(["--backends", "b"]).backends == ["b"]
        with pytest.raises(SystemExit):
            parser.parse_args(["--backends", "nope"])

    def test_first_dtype_is_the_default(self):
        """A benchmark that offers one dtype gets it without having to pass it."""
        parser = base_parser(backends=["a"], dtypes=("float64",))
        assert parser.parse_args([]).dtype == "float64"
        with pytest.raises(SystemExit):
            parser.parse_args(["--dtype", "float32"])

    def test_extends_with_its_own_axes(self):
        """The caller's sweep axes land in the same namespace as the shared ones."""
        parser = base_parser(backends=["a"])
        parser.add_argument("--wavelets", nargs="+", default=["haar"])
        args = parser.parse_args(["--wavelets", "db4", "db8", "--threads", "1"])
        assert args.wavelets == ["db4", "db8"]
        assert args.threads == 1
        assert args.device == "cpu"


class TestWithoutOptions:
    """Removing an option, and its value, from a command line."""

    def test_drops_an_option_and_its_value(self):
        """The child measures once; `--repeats` must not come along."""
        assert without_options(["--device", "cpu", "--repeats", "5"], {"--repeats"}) == ["--device", "cpu"]

    def test_drops_the_equals_form_too(self):
        """`--repeats=5` is the same flag written differently."""
        assert without_options(["--repeats=5", "--device", "cpu"], {"--repeats"}) == ["--device", "cpu"]

    def test_keeps_everything_else(self):
        """The sweep the child runs has to be the sweep that was asked for."""
        argv = ["--dims", "2", "3", "--threads", "1"]
        assert without_options(argv, {"--json"}) == argv

    def test_a_value_that_looks_like_a_flag_is_still_a_value(self):
        """Dropping a flag drops exactly one following token."""
        assert without_options(["--json", "--csv", "--threads", "1"], {"--json"}) == ["--threads", "1"]

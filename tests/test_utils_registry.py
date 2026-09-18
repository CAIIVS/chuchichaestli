# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for registry lookups."""

import pytest

from chuchichaestli.utils.registry import require


REGISTRY = {"b": 2, "a": 1, "c": 3}


def test_a_registered_name_returns_its_entry():
    """The lookup is the point; the error is the bonus."""
    assert require("a", REGISTRY) == 1


def test_an_unknown_name_lists_the_alternatives_in_order():
    """Sorted, so the message reads the same whatever the dict order."""
    with pytest.raises(
        ValueError, match=r"Unsupported: 'z'. Use one of \['a', 'b', 'c'\]"
    ):
        require("z", REGISTRY)


def test_the_context_says_which_position_was_wrong():
    """One message shape, whatever is being looked up."""
    with pytest.raises(ValueError, match="Unsupported optimizer: 'adamax'"):
        require("adamax", REGISTRY, "optimizer")


def test_it_holds_entries_of_any_kind():
    """Registries here hold classes, settings and handlers alike."""
    assert require(2, {1: "one", 2: "two"}) == "two"
    assert require("f", {"f": len})("abc") == 3


def test_a_fallback_resolves_what_the_registry_lacks():
    """An unregistered class is importable rather than refused."""
    assert require("z", REGISTRY, fallback=str.upper) == "Z"
    assert require("a", REGISTRY, fallback=str.upper) == 1


def test_a_plain_collection_validates_and_returns_the_name():
    """Not every registry maps to something; some only say what is allowed."""
    modes = frozenset({"min", "max"})
    assert require("min", modes, "mode") == "min"
    with pytest.raises(ValueError, match=r"Unsupported mode: 'mid'"):
        require("mid", modes, "mode")


def test_a_custom_message_replaces_the_default():
    """A file suffix reads better named by the verb than by 'Unsupported'."""
    suffix = ".ckpt"
    with pytest.raises(ValueError, match=r"Cannot read '.ckpt'; choose from"):
        require(
            suffix,
            {".pt": 1, ".safetensors": 2},
            message=lambda options: f"Cannot read '{suffix}'; choose from {options}.",
        )


def test_a_custom_message_is_only_built_on_failure():
    """It is an error path, so it must cost nothing when the lookup succeeds."""
    calls = []

    def message(options):
        calls.append(options)
        return "never used"

    assert require("a", REGISTRY, message=message) == 1
    assert calls == []

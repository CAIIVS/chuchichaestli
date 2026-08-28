# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the per-position argument helpers."""

import pytest
from chuchichaestli.utils import broadcast, broadcast_kwargs


@pytest.mark.parametrize(
    "value",
    [0.1, 3, "group", None, True],
    ids=["float", "int", "str", "none", "bool"],
)
def test_single_values_broadcast(value):
    """Test that anything that is not a list or tuple is broadcast to every position."""
    assert broadcast(value, 4) == (value,) * 4


@pytest.mark.parametrize(
    "seq", [(0.1, 0.2, 0.3), [0.1, 0.2, 0.3]], ids=["tuple", "list"]
)
def test_full_length_sequences_pass_through(seq):
    """Test that a sequence with one entry per position is kept as a tuple."""
    assert broadcast(seq, 3) == (0.1, 0.2, 0.3)


def test_masked_sequence_scatters_onto_marked_positions():
    """Test that a sequence as long as the mask lands on the marked positions."""
    mask = [False, False, True, True, True, False, False]
    assert broadcast((10, 20, 30), 7, mask=mask) == (10, 10, 10, 20, 30, 10, 10)


def test_full_length_wins_when_both_lengths_are_accepted():
    """Test that the full-length reading is preferred when the two counts coincide."""
    mask = [True, True, True]
    assert broadcast((1, 2, 3), 3, mask=mask) == (1, 2, 3)


def test_mask_is_ignored_without_marked_positions():
    """Test that an all-false mask leaves only the full length acceptable."""
    with pytest.raises(ValueError):
        broadcast((1, 2), 4, mask=[False] * 4)


def test_wrong_length_names_both_accepted_counts():
    """Test that the error message states the accepted counts and the positions."""
    mask = [False, True, True, False, False]
    with pytest.raises(ValueError) as excinfo:
        broadcast((1, 2, 3), 5, "attn_n_heads", mask, "[d0, d1, mid, u0, u1]")
    message = str(excinfo.value)
    assert "attn_n_heads" in message
    assert "5 values" in message
    assert "[d0, d1, mid, u0, u1]" in message
    assert "or 2" in message
    assert "got 3" in message


def test_args_expand_into_one_dict_per_position():
    """Test that a mapping of mixed single and per-position values is expanded."""
    expanded = broadcast_kwargs({"res_dropout": (0.0, 0.5), "res_groups": 8}, 2)
    assert expanded == [
        {"res_dropout": 0.0, "res_groups": 8},
        {"res_dropout": 0.5, "res_groups": 8},
    ]


def test_opaque_keys_are_never_expanded():
    """Test that opaque values reach every position unchanged, whatever their shape."""
    payload = (None, "rms")
    expanded = broadcast_kwargs(
        {"norm_type": payload, "n_heads": (1, 2, 3)}, 3, opaque=("norm_type",)
    )
    assert [d["norm_type"] for d in expanded] == [payload] * 3
    assert [d["n_heads"] for d in expanded] == [1, 2, 3]


def test_expanded_dicts_are_independent():
    """Test that each position gets its own dict rather than a shared object."""
    expanded = broadcast_kwargs({"res_groups": 8}, 3)
    expanded[0]["res_groups"] = 32
    assert [d["res_groups"] for d in expanded] == [32, 8, 8]

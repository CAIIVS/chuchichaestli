# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Consistency tests for the block registries."""

from chuchichaestli.models.blocks import (
    ATTENTION_BLOCK_MAP,
    BLOCK_MAP,
    EfficientViTBlock,
)


def bound_attention(cls) -> str | None:
    """Return the attention type a partialclass-generated block binds, if any."""
    return getattr(cls.__dict__.get("__init__"), "keywords", {}).get("attention")


def test_attention_block_map_is_a_subset_of_block_map():
    """Test that every attention block is a registered block of the same class."""
    assert {name: BLOCK_MAP.get(name) for name in ATTENTION_BLOCK_MAP} == dict(
        ATTENTION_BLOCK_MAP
    )


def test_attention_block_map_matches_the_bound_attention_types():
    """Test that the map cannot drift from the blocks that actually take attention."""
    derived = {name for name, cls in BLOCK_MAP.items() if bound_attention(cls)}
    derived.add("EfficientViTBlock")
    assert set(ATTENTION_BLOCK_MAP) == derived


def test_blocks_outside_the_map_take_no_attention():
    """Test that no block is left out of the map while still consuming attn_args."""
    for name, cls in BLOCK_MAP.items():
        if name not in ATTENTION_BLOCK_MAP:
            assert bound_attention(cls) is None
            assert cls is not EfficientViTBlock

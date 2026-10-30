# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for human-readable random names."""

import random
import re

import pytest
import torch

from chuchichaestli.utils import names
from chuchichaestli.utils.names import (
    ADJECTIVES,
    NOUNS,
    SlugGenerator,
    generate_slug,
    unique_name,
)


class Scripted(random.Random):
    """A random source that picks the given words, in order."""

    def __new__(cls, *picks: str):
        """Create the source; `random.Random.__new__` takes a seed only before 3.11.

        Args:
            picks: Words to return, handled by `__init__`.
        """
        return super().__new__(cls)

    def __init__(self, *picks: str):
        """Constructor.

        Args:
            picks: Words to return, one per call.
        """
        super().__init__(0)
        self.picks = list(picks)

    def choice(self, seq):
        """Return the next scripted word.

        Args:
            seq: The part drawn from, ignored.
        """
        return self.picks.pop(0)


CATEGORIES = (
    "QUALITIES",
    "COLORS",
    "SIZES",
    "ANIMALS",
    "CREATURES",
    "NATURE",
    "COSMOS",
)


@pytest.mark.parametrize("category", CATEGORIES)
def test_a_category_is_sorted_lowercase_ascii(category):
    """A slug is an identifier, so every word must be safe in one."""
    words = getattr(names, category)
    assert list(words) == sorted(words)
    assert all(re.fullmatch("[a-z]+", word) for word in words)


@pytest.mark.parametrize("part", [ADJECTIVES, NOUNS], ids=["adjectives", "nouns"])
def test_no_word_repeats_across_the_categories_of_a_part(part):
    """A word in two categories would be drawn twice as often."""
    assert len(set(part)) == len(part)


def test_the_two_parts_share_no_word():
    """Disjoint parts make every pair a distinct slug."""
    assert not set(ADJECTIVES) & set(NOUNS)


def test_combinations_count_the_pairs():
    """Every adjective pairs with every noun."""
    assert SlugGenerator().combinations == len(ADJECTIVES) * len(NOUNS)
    assert SlugGenerator().combinations > 150_000


def test_a_seeded_source_draws_the_same_slug():
    """Tests and examples can pin a name."""
    assert generate_slug(rand=random.Random(7)) == generate_slug(rand=random.Random(7))
    assert SlugGenerator(rand=random.Random(7)).slug().count("-") == 1
    assert SlugGenerator(rand=random.Random(7)).slug("_").count("_") == 1


def test_a_slug_is_an_adjective_and_a_noun():
    """The order reads as a phrase."""
    adjective, noun = SlugGenerator(rand=random.Random(3)).generate()
    assert adjective in ADJECTIVES
    assert noun in NOUNS


def test_a_slug_never_repeats_a_word():
    """`echo-echo` reads as a typo, so a repeat is drawn again."""
    generator = SlugGenerator(
        (("echo", "dawn"), ("echo",)), Scripted("echo", "echo", "dawn", "echo")
    )
    assert generator.generate() == ["dawn", "echo"]


def test_parts_that_cannot_avoid_a_repeat_are_refused():
    """Drawing forever would hang the run that asked for a name."""
    with pytest.raises(ValueError, match="repeated word"):
        SlugGenerator((("echo",), ("echo",)), attempts=3).generate()


@pytest.mark.parametrize("parts", [(), (("a",), ())], ids=["none", "empty"])
def test_a_generator_needs_parts_to_draw_from(parts):
    """An empty part has no word to give."""
    with pytest.raises(ValueError, match="at least one part"):
        SlugGenerator(parts)


def test_naming_leaves_the_seeded_generators_alone():
    """A run's name must not change the run's randomness."""
    random.seed(1234)
    torch.manual_seed(1234)
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    generate_slug()
    unique_name({"warm-badger"})
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_a_unique_name_avoids_those_taken():
    """A clash is drawn again before it is numbered."""
    generator = SlugGenerator(rand=Scripted("warm", "badger", "calm", "otter"))
    assert unique_name({"warm-badger"}, generator) == "calm-otter"


def test_repeated_clashes_take_the_lowest_free_number():
    """Only a name that keeps clashing gets a suffix."""
    generator = SlugGenerator((("warm",), ("badger",)))
    taken = {"warm-badger", "warm-badger-2"}
    assert unique_name(taken, generator, attempts=3) == "warm-badger-3"
    assert (
        unique_name({"warm_badger"}, generator, attempts=1, sep="_") == "warm_badger_2"
    )


def test_a_free_name_needs_no_number():
    """Nothing taken, nothing appended."""
    generator = SlugGenerator((("warm",), ("badger",)))
    assert unique_name((), generator) == "warm-badger"

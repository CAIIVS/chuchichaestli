# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for dividing a dataset into disjoint parts."""

import pytest

from chuchichaestli.data import split_dataset


def test_the_parts_cover_the_dataset_exactly():
    """No sample may be lost or counted twice."""
    parts = split_dataset(list(range(20)), {"train": 0.8, "val": 0.2})
    covered = sorted(parts["train"].indices + parts["val"].indices)
    assert covered == list(range(20))
    assert set(parts["train"].indices).isdisjoint(parts["val"].indices)


def test_the_same_seed_gives_the_same_parts():
    """A resumed run must validate on the same samples."""
    kwargs = {"fractions": [0.7, 0.3], "seed": 42}
    first = [part.indices for part in split_dataset(list(range(50)), **kwargs)]
    assert [part.indices for part in split_dataset(list(range(50)), **kwargs)] == first
    other = [
        part.indices for part in split_dataset(list(range(50)), [0.7, 0.3], seed=1)
    ]
    assert other != first


def test_a_mapping_comes_back_as_a_mapping():
    """The shape in is the shape out, so callers can index by name."""
    assert sorted(split_dataset(list(range(10)), {"a": 0.5, "b": 0.5})) == ["a", "b"]
    assert len(split_dataset(list(range(10)), [0.5, 0.5])) == 2


def test_without_shuffle_the_parts_are_contiguous():
    """For data already ordered the way the split should respect."""
    parts = split_dataset(list(range(10)), [0.6, 0.4], shuffle=False)
    assert parts[0].indices == list(range(6))
    assert parts[1].indices == list(range(6, 10))


@pytest.mark.parametrize(
    ("total", "fractions", "expected"),
    [
        (10, [0.5, 0.5], [5, 5]),
        (10, [0.8, 0.2], [8, 2]),
        (7, [1 / 3, 1 / 3, 1 / 3], [3, 2, 2]),
    ],
)
def test_the_parts_always_sum_to_the_dataset(total, fractions, expected):
    """Torch spreads the remainder; nothing may be lost."""
    parts = split_dataset(list(range(total)), fractions)
    assert sorted(len(p) for p in parts) == sorted(expected)
    assert sum(len(p) for p in parts) == total


def test_a_negative_fraction_is_refused():
    """It would ask for a part with fewer than no samples."""
    with pytest.raises(ValueError, match="must be positive"):
        split_dataset(list(range(10)), [0.5, -0.5])


def test_a_part_that_would_be_empty_is_refused():
    """Better than handing back a dataset nothing can draw from."""
    with pytest.raises(ValueError, match="empty part"):
        split_dataset(list(range(3)), [0.98, 0.01, 0.01])

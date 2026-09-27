# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for derivation-based randomness."""

import subprocess
import sys

import torch

from chuchichaestli.utils.rng import (
    WorkerSeeder,
    capture_rng_state,
    derive_seed,
    rng_generator,
    restore_rng_state,
    seed_ambient,
)


def test_derive_seed_is_a_pure_function_of_seed_and_path():
    """Same inputs give the same seed; different inputs do not."""
    assert derive_seed(42, "a/b") == derive_seed(42, "a/b")
    assert derive_seed(42, "a/b") != derive_seed(42, "a/c")
    assert derive_seed(42, "a/b") != derive_seed(43, "a/b")


def test_derive_seed_is_stable_across_processes():
    """It must not use the salted built-in hash, or resume would diverge."""
    here = derive_seed(42, "program/0:train/data/epoch=3")
    code = (
        "from chuchichaestli.utils.rng import derive_seed;"
        "print(derive_seed(42, 'program/0:train/data/epoch=3'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert int(out.stdout.strip()) == here


def test_rng_generator_is_reproducible_per_position():
    """Two generators for the same position draw the same numbers."""
    first = torch.randn(4, generator=rng_generator(7, "x"))
    again = torch.randn(4, generator=rng_generator(7, "x"))
    other = torch.randn(4, generator=rng_generator(7, "y"))
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_ambient_capture_round_trip():
    """Restoring captured state reproduces the next draws exactly."""
    seed_ambient(3)
    state = capture_rng_state()
    expected = torch.randn(5)
    restore_rng_state(state)
    assert torch.equal(torch.randn(5), expected)


def test_worker_seeder_is_picklable():
    """It must survive pickling into spawn and forkserver workers."""
    import pickle

    seeder = WorkerSeeder(11, "program/0:train/data")
    revived = pickle.loads(pickle.dumps(seeder))
    assert revived == seeder
    revived(0)
    first = torch.randn(2)
    revived(0)
    assert torch.equal(torch.randn(2), first)


def test_worker_seeds_differ_per_worker():
    """Workers must not all draw the same stream."""
    seeder = WorkerSeeder(11, "d")
    seeder(0)
    zero = torch.randn(2)
    seeder(1)
    assert not torch.equal(torch.randn(2), zero)

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for the windowing stack.

Exercises `ZipDataset` -> `SlidingWindowBatchSampler` -> `DataLoader` +
`SlidingWindowCollate` together. The components pass their own unit tests in
isolation; the defects these cover only appear once they are composed.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from chuchichaestli.data.batching import SlidingWindowBatchSampler
from chuchichaestli.data.collate import SlidingWindowCollate
from chuchichaestli.data.numpy import NumpyDataset
from chuchichaestli.data.zip import ZipDataset

LENGTHS = (20, 12)
WINDOW, HORIZON, BATCH = 4, 1, 2
SPAN = WINDOW + HORIZON


@pytest.fixture
def paired_files(tmp_path):
    """Two files per stream, of differing length, as `(main, side)` paths."""
    rng = np.random.default_rng(0)
    main, side = [], []
    for i, length in enumerate(LENGTHS):
        m = tmp_path / f"main_{i}.npy"
        s = tmp_path / f"side_{i}.npy"
        np.save(m, rng.standard_normal((length, 8, 6)).astype(np.float32))
        np.save(s, rng.standard_normal((length, 3)).astype(np.float32))
        main.append(str(m))
        side.append(str(s))
    return main, side


@pytest.fixture
def zipped(paired_files):
    """A two-stream `ZipDataset` over those files."""
    main, side = paired_files
    return ZipDataset(
        NumpyDataset(main, cache="16M"),
        NumpyDataset(side, cache="16M"),
        zip_as={"main": 0, "side": 1},
    )


def windows_of(batch):
    """Split one flattened sampler batch back into its windows."""
    return [batch[i : i + SPAN] for i in range(0, len(batch), SPAN)]


class TestBoundaries:
    """Sequence boundaries must survive the zip."""

    def test_zip_reports_the_constituent_offsets(self, zipped):
        """Without this the sampler cannot tell the two files apart."""
        assert zipped._file_offsets == [0, LENGTHS[0], sum(LENGTHS)]

    def test_no_window_straddles_a_file(self, zipped):
        """A window spanning two files would splice unrelated sequences."""
        sampler = SlidingWindowBatchSampler(
            zipped, window_size=WINDOW, horizon=HORIZON, batch_size=BATCH
        )
        cut = LENGTHS[0]
        for batch in sampler:
            for window in windows_of(batch):
                assert max(window) < cut or min(window) >= cut

    def test_derived_boundaries_match_explicit_ones(self, zipped):
        """Deriving must give exactly what passing them by hand gives."""
        kwargs = {"window_size": WINDOW, "horizon": HORIZON, "batch_size": BATCH}
        derived = SlidingWindowBatchSampler(zipped, **kwargs)
        explicit = SlidingWindowBatchSampler(
            zipped, boundaries=[0, LENGTHS[0], sum(LENGTHS)], **kwargs
        )
        assert list(derived) == list(explicit)


class TestBatchAssembly:
    """Shapes and the input/target split through a real DataLoader."""

    @staticmethod
    def _loader(zipped, **collate_kwargs):
        """A loader wiring the sampler and collate to the same geometry."""
        sampler = SlidingWindowBatchSampler(
            zipped, window_size=WINDOW, horizon=HORIZON, batch_size=BATCH
        )
        collate = SlidingWindowCollate.from_sampler(sampler, **collate_kwargs)
        return DataLoader(zipped, batch_sampler=sampler, collate_fn=collate)

    def test_window_axis_is_restored(self, zipped):
        """The flattened batch becomes (batch, window, ...) per entry."""
        batch = next(iter(self._loader(zipped)))
        assert batch["main"].shape == (BATCH, WINDOW, 8, 6)
        assert batch["side"].shape == (BATCH, WINDOW, 3)

    def test_horizon_is_split_off(self, zipped):
        """Trailing samples become the target of each entry."""
        batch = next(iter(self._loader(zipped)))
        assert batch["main_target"].shape == (BATCH, HORIZON, 8, 6)
        assert batch["side_target"].shape == (BATCH, HORIZON, 3)

    def test_every_batch_is_full(self, zipped):
        """The sampler and collate agree on the span for all batches."""
        for batch in self._loader(zipped):
            assert batch["main"].shape[1] == WINDOW

    def test_workers_attach_to_the_shared_cache(self, zipped):
        """Workers must fill the parent's cache, not private copies."""
        sampler = SlidingWindowBatchSampler(
            zipped, window_size=WINDOW, horizon=HORIZON, batch_size=BATCH
        )
        collate = SlidingWindowCollate.from_sampler(sampler)
        loader = DataLoader(
            zipped, batch_sampler=sampler, collate_fn=collate, num_workers=2
        )
        assert sum(1 for _ in loader) == len(sampler)
        assert zipped.datasets[0].n_cached > 0


class TestPerKeyTransform:
    """A transform aimed at one stream must leave the other alone."""

    def test_only_the_named_entry_is_transformed(self, zipped):
        """One projection rarely suits every entry of a zipped sample."""
        sampler = SlidingWindowBatchSampler(
            zipped, window_size=WINDOW, horizon=HORIZON, batch_size=BATCH
        )
        collate = SlidingWindowCollate.from_sampler(
            sampler, transform={"main": lambda t: t[..., :2]}
        )
        batch = next(
            iter(DataLoader(zipped, batch_sampler=sampler, collate_fn=collate))
        )
        assert batch["main"].shape == (BATCH, WINDOW, 8, 2)
        assert batch["side"].shape == (BATCH, WINDOW, 3)

    def test_a_bare_callable_still_reaches_both(self, zipped):
        """The single-callable form is unchanged."""
        sampler = SlidingWindowBatchSampler(
            zipped, window_size=WINDOW, horizon=HORIZON, batch_size=BATCH
        )
        collate = SlidingWindowCollate.from_sampler(sampler, transform=torch.zeros_like)
        batch = next(
            iter(DataLoader(zipped, batch_sampler=sampler, collate_fn=collate))
        )
        assert not batch["main"].any()
        assert not batch["side"].any()

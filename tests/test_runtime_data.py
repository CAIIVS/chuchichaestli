# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for deterministically ordered, resumable passes over a dataset."""

import pytest
import torch

from chuchichaestli.data.batching import HierarchicalBatchSampler
from chuchichaestli.runtime import C3liDataError, Context, DataManager, Local


class Counting(torch.utils.data.Dataset):
    """A dataset whose samples are their own indices."""

    def __init__(self, n: int):
        """Constructor.

        Args:
            n: How many samples.
        """
        self.n = n

    def __len__(self) -> int:
        """Number of samples."""
        return self.n

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the sample at an index.

        Args:
            index: Which sample.
        """
        return torch.tensor(index)


class Ranks:
    """A stand-in topology of a given size."""

    def __init__(self, rank: int, world_size: int):
        """Constructor.

        Args:
            rank: Index of this process.
            world_size: How many processes.
        """
        self.rank = rank
        self.world_size = world_size


def ctx(seed: int = 7, path: str = "program/0:train") -> Context:
    """Build a context to derive orders from.

    Args:
        seed: Root seed of the run.
        path: Stage path the stream belongs to.
    """
    return Context(path, seed)


def drawn(stream: DataManager, context: Context, epoch: int = 0, seek: int = 0):
    """Return the sample indices a stream yields, batch by batch.

    Args:
        stream: The stream to draw from.
        context: Context the order derives from.
        epoch: Which pass over the dataset.
        seek: Batches to skip.
    """
    return [batch.tolist() for batch in stream.open(context, epoch, seek)]


def test_the_same_seed_gives_the_same_order():
    """Order is a function of the seed and the position, nothing else."""
    stream = DataManager(Counting(10), batch_size=3)
    assert stream.plan(ctx()) == stream.plan(ctx())
    assert stream.plan(ctx(seed=8)) != stream.plan(ctx(seed=7))


def test_each_epoch_draws_a_different_order():
    """A pass that repeated itself would not be shuffling."""
    stream = DataManager(Counting(10), batch_size=3)
    assert stream.plan(ctx(), epoch=0) != stream.plan(ctx(), epoch=1)
    assert stream.plan(ctx(), epoch=1) == stream.plan(ctx(), epoch=1)


def test_two_stages_at_different_paths_draw_differently():
    """The order derives from the position, so streams do not move together."""
    stream = DataManager(Counting(10), batch_size=3)
    here = stream.plan(ctx(path="program/0:train"))
    there = stream.plan(ctx(path="program/1:finetune"))
    assert here != there


def test_every_sample_appears_exactly_once():
    """Shuffling must permute, not resample."""
    plan = DataManager(Counting(10), batch_size=3).plan(ctx())
    assert sorted(index for batch in plan for index in batch) == list(range(10))


def test_a_short_final_batch_is_kept_by_default():
    """Correct micro-batch weighting subsumes the need to discard it."""
    plan = DataManager(Counting(10), batch_size=3).plan(ctx())
    assert [len(batch) for batch in plan] == [3, 3, 3, 1]


def test_drop_last_discards_it():
    """For callers who want every batch the same size."""
    plan = DataManager(Counting(10), batch_size=3, drop_last=True).plan(ctx())
    assert [len(batch) for batch in plan] == [3, 3, 3]


def test_without_shuffle_the_order_is_the_datasets():
    """An eval pass has no reason to permute."""
    plan = DataManager(Counting(6), batch_size=2, shuffle=False).plan(ctx())
    assert plan == [[0, 1], [2, 3], [4, 5]]


def test_seek_matches_the_tail_of_an_uninterrupted_epoch():
    """Mid-epoch resume slices the plan rather than replaying the loader."""
    stream = DataManager(Counting(12), batch_size=3)
    whole = drawn(stream, ctx())
    assert drawn(stream, ctx(), seek=2) == whole[2:]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_what_is_drawn_does_not_depend_on_the_worker_count(num_workers):
    """Workers seed themselves from the stream's position, not from chance."""
    alone = drawn(DataManager(Counting(12), batch_size=3), ctx())
    assert (
        drawn(DataManager(Counting(12), batch_size=3, num_workers=num_workers), ctx())
        == alone
    )


def test_a_sampler_supplies_the_batches_and_the_stream_their_order():
    """Variable-size batches keep their contents; only the order is ours."""
    batches = [[0, 1], [2], [3, 4, 5], [6]]
    stream = DataManager(Counting(7), batches=HierarchicalBatchSampler(batches))
    plan = stream.plan(ctx())
    assert sorted(plan) == sorted(batches)
    assert any(plan != stream.plan(ctx(), epoch=e) for e in (1, 2, 3))


def test_a_sampler_plan_survives_a_seek():
    """The same slicing applies whatever produced the batches."""
    batches = [[0, 1], [2], [3, 4, 5], [6]]
    stream = DataManager(
        Counting(7), batches=HierarchicalBatchSampler(batches), shuffle=False
    )
    assert stream.plan(ctx()) == batches
    assert drawn(stream, ctx(), seek=2) == [[3, 4, 5], [6]]


def test_a_self_shuffling_sampler_is_refused():
    """It draws from the global generator, so a resume would reorder."""
    sampler = HierarchicalBatchSampler([[0, 1], [2, 3]], shuffle=True)
    with pytest.raises(C3liDataError, match="shuffle=True"):
        DataManager(Counting(4), batches=sampler)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"num_workers": -1}])
def test_an_unusable_stream_is_refused(kwargs):
    """Both would fail later and less clearly."""
    with pytest.raises(ValueError):
        DataManager(Counting(4), **kwargs)


def test_sharding_partitions_without_overlap():
    """Every batch goes to exactly one rank when the counts divide."""
    stream = DataManager(Counting(12), batch_size=3)
    plan = stream.plan(ctx())
    shards = [stream.shard(plan, Ranks(rank, 2)) for rank in range(2)]
    assert sorted(shards[0] + shards[1]) == sorted(plan)


def test_every_rank_draws_the_same_number_of_batches():
    """An uneven share is a deadlock, not a rounding error."""
    stream = DataManager(Counting(10), batch_size=3)
    plan = stream.plan(ctx())
    assert len(plan) == 4
    counts = {len(stream.shard(plan, Ranks(rank, 3))) for rank in range(3)}
    assert counts == {2}


def test_padding_wraps_from_the_front():
    """A repeat is better than a rank running dry mid-epoch."""
    stream = DataManager(Counting(10), batch_size=3, shuffle=False)
    plan = stream.plan(ctx())
    seen = [batch for rank in range(3) for batch in stream.shard(plan, Ranks(rank, 3))]
    assert len(seen) == 6
    assert seen.count(plan[0]) == 2


def test_a_single_process_is_handed_the_whole_plan():
    """Sharding must cost nothing when there is nothing to share with."""
    stream = DataManager(Counting(10), batch_size=3)
    plan = stream.plan(ctx())
    assert stream.shard(plan, Local(device="cpu")) is plan


def test_the_sharded_plan_is_this_processs_share():
    """What a loop needs before it draws: how many batches, and how big."""
    stream = DataManager(Counting(10), batch_size=3)
    assert stream.sharded_plan(ctx()) == stream.plan(ctx())
    assert len(stream.sharded_plan(ctx())) == 4
    assert [len(batch) for batch in stream.sharded_plan(ctx())] == [3, 3, 3, 1]


def test_a_stream_can_draw_from_one_part_of_a_dataset():
    """Train and eval streams over one source must not overlap."""
    shared = {"split": {"train": 0.8, "val": 0.2}, "split_seed": 42, "batch_size": 4}
    train = DataManager(Counting(20), part="train", **shared)
    val = DataManager(Counting(20), part="val", **shared)
    assert len(train.dataset) == 16
    assert len(val.dataset) == 4
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)


def test_the_same_split_seed_gives_the_same_part():
    """Two stages splitting one dataset must agree on where the line is."""
    shared = {"split": [0.7, 0.3], "split_seed": 3, "batch_size": 2}
    first = DataManager(Counting(20), part=1, **shared).dataset.indices
    assert DataManager(Counting(20), part=1, **shared).dataset.indices == first
    other = DataManager(
        Counting(20), part=1, split=[0.7, 0.3], split_seed=4, batch_size=2
    )
    assert other.dataset.indices != first


def test_a_part_the_split_does_not_hold_says_what_it_does():
    """A typo in a stage config should not be a silent empty stream."""
    with pytest.raises(C3liDataError, match=r"holds \['train', 'val'\]"):
        DataManager(Counting(20), split={"train": 0.8, "val": 0.2}, part="test")


def test_a_split_without_a_part_is_refused():
    """Half a request is a mistake, not a default."""
    with pytest.raises(ValueError, match="needs a part"):
        DataManager(Counting(20), split=[0.5, 0.5])
    with pytest.raises(ValueError, match="needs a part"):
        DataManager(Counting(20), part="train")


def test_the_loader_options_reach_the_loader():
    """They are passed through, so a wrong one must fail here not later."""
    stream = DataManager(
        Counting(12),
        batch_size=3,
        num_workers=2,
        pin_memory=False,
        timeout=5,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    assert [batch.tolist() for batch in stream.open(ctx())] == drawn(
        DataManager(Counting(12), batch_size=3), ctx()
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"prefetch_factor": 2}, {"persistent_workers": True}],
    ids=lambda k: next(iter(k)),
)
def test_a_worker_option_without_workers_is_refused(kwargs):
    """Torch would raise on the first draw; this raises where it is written."""
    with pytest.raises(ValueError, match="needs workers"):
        DataManager(Counting(12), batch_size=3, num_workers=0, **kwargs)


def test_a_negative_timeout_is_refused():
    """It would mean waiting a negative time for a batch."""
    with pytest.raises(ValueError, match="negative timeout"):
        DataManager(Counting(12), timeout=-1)


def test_the_loader_generator_is_derived_when_none_is_pinned():
    """Torch seeds its workers from it, so it must not be left to chance."""
    stream = DataManager(Counting(12), batch_size=3)
    first = stream._generator(ctx(), "data/epoch=0").initial_seed()
    assert stream._generator(ctx(), "data/epoch=0").initial_seed() == first
    assert stream._generator(ctx(seed=8), "data/epoch=0").initial_seed() != first
    assert stream._generator(ctx(), "data/epoch=1").initial_seed() != first


def test_a_pinned_seed_replaces_the_runs_but_still_derives():
    """A generator fixed for the whole run would repeat every epoch."""
    stream = DataManager(Counting(12), batch_size=3, seed=1234)
    first = stream._generator(ctx(), "data/epoch=0").initial_seed()
    assert stream._generator(ctx(seed=8), "data/epoch=0").initial_seed() == first
    assert stream._generator(ctx(), "data/epoch=1").initial_seed() != first
    assert stream._generator(ctx(), "data/epoch=0").initial_seed() == first

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the training loop: one execute is one optimizer step."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.runtime import (
    Alternating,
    Phase,
    Program,
    Repeat,
    Context,
    DataManager,
    EventType,
    Finetune,
    Runtime,
    Signal,
    Train,
)
from chuchichaestli.runtime.runtime import C3liProgramError
from chuchichaestli.training import OptimSpec


def linear(out: int = 1, fill: float | None = None) -> nn.Module:
    """Build a small model, optionally with a known weight.

    Args:
        out: Output features.
        fill: Value every weight starts at, or `None` for the default.
    """
    model = nn.Linear(3, out, bias=False)
    if fill is not None:
        with torch.no_grad():
            model.weight.fill_(fill)
    return model


def ramp(n: int = 5) -> TensorDataset:
    """Build a dataset whose inputs count upwards.

    Args:
        n: Number of samples.
    """
    x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    return TensorDataset(x, torch.zeros(n, 1))


def gradients_after_one_step(
    batch_size: int, accumulate: int, **kwargs
) -> torch.Tensor:
    """Run one step at zero learning rate and return the gradient it left.

    Args:
        batch_size: Samples per micro-batch.
        accumulate: Micro-batches per step.
        kwargs: Passed to `Train`.
    """
    model = linear(fill=0.5)
    data = DataManager(ramp(), batch_size=batch_size, shuffle=False)
    stage = Train(
        "t", model=model, data=data, steps=1, accumulate=accumulate, lr=0.0, **kwargs
    )
    ctx = Context("t")
    stage.enter(ctx)
    stage.execute(ctx)
    return model.weight.grad.clone()


def test_a_tiny_model_converges():
    """Training reduces the loss it was given."""
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = x @ torch.tensor([[1.0], [2.0], [-1.0], [0.5]]) + 0.3
    model = nn.Linear(4, 1)

    def loss_now() -> float:
        """Return the model's loss on its own device."""
        device = next(model.parameters()).device
        out = model(x.to(device))
        return float(nn.functional.mse_loss(out, y.to(device)).detach())

    before = loss_now()
    Runtime(
        Train(
            "fit",
            model=model,
            data=TensorDataset(x, y),
            loss=nn.MSELoss(),
            epochs=20,
            lr=0.05,
        ),
        hooks=(),
    ).run()
    assert loss_now() < before / 5


def test_accumulation_matches_one_larger_batch():
    """Micro-batches weighted by sample count reproduce one unsplit pass."""
    whole = gradients_after_one_step(5, 1, loss=nn.MSELoss())
    split = gradients_after_one_step(2, 3, loss=nn.MSELoss())
    assert torch.allclose(whole, split, atol=1e-5)


def test_the_even_share_shortcut_would_fail_that():
    """Weighting a short final micro-batch by `1/k` gives a different gradient."""
    whole = gradients_after_one_step(5, 1, loss=nn.MSELoss())
    naive = linear(fill=0.5)
    x, y = ramp().tensors
    for part in (slice(0, 2), slice(2, 4), slice(4, 5)):
        (nn.MSELoss()(naive(x[part]), y[part]) / 3).backward()
    assert not torch.allclose(whole, naive.weight.grad, atol=1e-5)


def test_accumulation_matches_for_a_summed_objective():
    """Under `reduction="sum"` the micro-batches simply add up."""
    whole = gradients_after_one_step(
        5, 1, loss=nn.MSELoss(reduction="sum"), reduction="sum"
    )
    split = gradients_after_one_step(
        2, 3, loss=nn.MSELoss(reduction="sum"), reduction="sum"
    )
    assert torch.allclose(whole, split, atol=1e-5)


def test_one_execute_is_one_optimizer_step():
    """Progress counts steps, not micro-batches."""
    stage = Train(
        "t",
        model=linear(),
        data=DataManager(ramp(6), batch_size=1, shuffle=False),
        loss=nn.MSELoss(),
        steps=2,
        accumulate=3,
    )
    ctx = Context("t")
    stage.enter(ctx)
    stage.execute(ctx)
    assert stage.progress().step == 1
    assert stage.progress().samples == 3


def test_steps_stop_the_stage():
    """A step budget ends the stage even mid-epoch."""
    stage = Train(
        "t",
        model=linear(),
        data=DataManager(ramp(10), batch_size=1, shuffle=False),
        loss=nn.MSELoss(),
        steps=3,
    )
    Runtime(stage, hooks=()).run()
    assert stage.progress().global_step == 3
    assert stage.progress().done


def test_epochs_stop_the_stage():
    """An epoch budget ends the stage at an epoch boundary."""
    stage = Train(
        "t",
        model=linear(),
        data=DataManager(ramp(4), batch_size=2, shuffle=False),
        loss=nn.MSELoss(),
        epochs=3,
    )
    Runtime(stage, hooks=()).run()
    assert stage.progress().epoch == 3
    assert stage.progress().global_step == 6


def test_a_train_without_a_budget_is_rejected_before_any_compute():
    """Neither epochs nor steps would never stop."""
    runtime = Runtime(
        Train("t", model=linear(), data=ramp(), loss=nn.MSELoss()), hooks=()
    )
    with pytest.raises(C3liProgramError, match="neither epochs nor steps"):
        runtime.check()


def test_a_train_needs_something_to_minimise():
    """Neither a loss nor an objective is refused at construction."""
    with pytest.raises(ValueError, match="neither a loss nor an objective"):
        Train("t", model=linear(), data=ramp(), epochs=1)


def test_epoch_events_bracket_each_pass():
    """Every epoch announces its start and its end."""
    seen: list[str] = []

    class Watching:
        """Records the events it sees."""

        def on(self, event):
            """Record one event.

            Args:
                event: What the runtime just did.
            """
            seen.append(event.type.value)
            return Signal.GO

    Runtime(
        Train(
            "t",
            model=linear(),
            data=DataManager(ramp(4), batch_size=2, shuffle=False),
            loss=nn.MSELoss(),
            epochs=2,
        ),
        hooks=(Watching(),),
    ).run()
    assert seen.count("epoch.began") == 2
    assert seen.count("epoch.ended") == 2
    assert seen.count("step.ended") == 4


def test_the_step_payload_carries_the_loss():
    """A step reports what it minimised."""
    payloads: list[dict] = []

    class Watching:
        """Records step payloads."""

        def on(self, event):
            """Record one event.

            Args:
                event: What the runtime just did.
            """
            if event.type is EventType.STEP_ENDED:
                payloads.append(event.payload)
            return Signal.GO

    Runtime(
        Train(
            "t",
            model=linear(),
            data=DataManager(ramp(2), batch_size=2, shuffle=False),
            loss=nn.MSELoss(),
            steps=1,
        ),
        hooks=(Watching(),),
    ).run()
    assert payloads and "loss" in payloads[0]


def test_a_reduced_precision_forward_runs_in_that_dtype():
    """Autocast applies on CPU with bfloat16."""
    seen: list[torch.dtype] = []

    class Noting(nn.Module):
        """A loss that notes the dtype it was handed."""

        def forward(self, output, target):
            """Record the dtype and reduce.

            Args:
                output: What the model produced.
                target: What it is compared against.
            """
            seen.append(output.dtype)
            return nn.functional.mse_loss(output.float(), target.float())

    stage = Train(
        "t",
        model=linear(),
        data=DataManager(ramp(2), batch_size=2, shuffle=False),
        loss=Noting(),
        steps=1,
        precision=torch.bfloat16,
    )
    Runtime(stage, hooks=(), device="cpu").run()
    assert seen == [torch.bfloat16]


def test_finetune_is_a_train_preset():
    """Finetune trains with a gentler default learning rate."""
    assert issubclass(Finetune, Train)
    stage = Finetune("refine", model=linear(), data=ramp(), loss=nn.MSELoss(), steps=1)
    assert stage.lr == 1e-5


def test_a_bare_dataset_is_wrapped_in_a_manager():
    """A stage may be handed a dataset instead of a manager."""
    assert isinstance(DataManager.from_source(ramp()), DataManager)
    manager = DataManager(ramp())
    assert DataManager.from_source(manager) is manager
    with pytest.raises(ValueError, match="needs data"):
        DataManager.from_source(None)


def test_optimizer_state_rides_in_the_stage_state():
    """A resumed stage restores its optimizer, not just its counters."""
    stage = Train(
        "t",
        model=linear(),
        data=DataManager(ramp(4), batch_size=2, shuffle=False),
        loss=nn.MSELoss(),
        steps=2,
        optim=None,
    )
    Runtime(stage, hooks=()).run()
    state = stage.state_dict()
    assert "progress" in state
    assert "optim" in state


def test_a_bare_dataset_takes_the_stage_batch_size():
    """The knob that defines a step sits beside `accumulate`."""
    stage = Train(
        "t", model=linear(), data=ramp(8), batch_size=4, loss=nn.MSELoss(), steps=1
    )
    ctx = Context("t")
    stage.enter(ctx)
    stage.execute(ctx)
    assert stage.progress().samples == 4


def test_batch_size_multiplies_with_accumulate():
    """Together they say what one optimizer step consumes."""
    stage = Train(
        "t",
        model=linear(),
        data=ramp(8),
        batch_size=2,
        accumulate=3,
        loss=nn.MSELoss(),
        steps=1,
    )
    ctx = Context("t")
    stage.enter(ctx)
    stage.execute(ctx)
    assert stage.progress().samples == 6


def test_a_built_manager_keeps_its_own_batch_size():
    """Giving both would leave one silently ignored, so it is refused."""
    with pytest.raises(ValueError, match="set them in one place"):
        DataManager.from_source(DataManager(ramp(8), batch_size=8), batch_size=16)


def test_a_bare_dataset_still_defaults_to_one():
    """The shorthand without a batch size is unchanged."""
    assert DataManager.from_source(ramp()).batch_size == 1


def test_from_source_forwards_every_option():
    """As a constructor it takes whatever `DataManager` takes."""
    manager = DataManager.from_source(
        ramp(8), batch_size=4, shuffle=False, drop_last=True
    )
    assert (manager.batch_size, manager.shuffle, manager.drop_last) == (4, False, True)


def test_effective_batch_reports_what_a_step_consumes():
    """It multiplies the batch size, the accumulation and the processes."""
    stage = Train(
        "t",
        model=linear(),
        data=ramp(16),
        batch_size=4,
        accumulate=2,
        loss=nn.MSELoss(),
        steps=1,
    )
    assert stage.effective_batch is None
    stage.enter(Context("t"))
    assert stage.effective_batch == 8


def test_effective_batch_counts_every_process():
    """Adding ranks raises it, which is what makes it worth inspecting."""

    class TwoRanks:
        """A stand-in topology of two processes."""

        rank, local_rank, world_size = 0, 0, 2
        device = torch.device("cpu")
        is_main = True

        def reduce(self, value, op="mean"):
            """Return the value unchanged.

            Args:
                value: Tensor held by this process.
                op: Ignored.
            """
            return value

        def barrier(self):
            """Do nothing."""

        def wrap(self, module):
            """Return the module unchanged.

            Args:
                module: Module to wrap.
            """
            return module

        def broadcast(self, value):
            """Return the value unchanged.

            Args:
                value: This process's candidate value.
            """
            return value

    stage = Train(
        "t", model=linear(), data=ramp(16), batch_size=4, loss=nn.MSELoss(), steps=1
    )
    stage.enter(Context("t", topology=TwoRanks()))
    assert stage.effective_batch == 8


def test_effective_batch_follows_a_built_manager():
    """A manager's own batch size is what counts, not the stage's."""
    stage = Train(
        "t",
        model=linear(),
        data=DataManager(ramp(16), batch_size=8, shuffle=False),
        accumulate=2,
        loss=nn.MSELoss(),
        steps=1,
    )
    stage.enter(Context("t"))
    assert stage.effective_batch == 16


def test_an_update_and_its_settings_cannot_both_be_given():
    """A built update carries its own policy, so the knobs would be ignored."""
    with pytest.raises(ValueError, match=r"\['clip'\] configure an update"):
        Train(
            "t",
            model=linear(),
            data=ramp(),
            loss=nn.MSELoss(),
            steps=1,
            update=Alternating(("disc", "gen")),
            clip=1.0,
        )
    with pytest.raises(ValueError, match=r"\['precision', 'reduction'\]"):
        Train(
            "t",
            model=linear(),
            data=ramp(),
            loss=nn.MSELoss(),
            steps=1,
            update=Alternating(("disc", "gen")),
            precision=torch.bfloat16,
            reduction="sum",
        )


def test_the_settings_reach_the_default_update():
    """Without an update of its own the stage configures the one it builds."""
    stage = Train(
        "t",
        model=linear(),
        data=ramp(),
        loss=nn.MSELoss(),
        steps=1,
        clip=1.0,
        clip_mode="value",
        reduction="sum",
        precision=torch.bfloat16,
    )
    assert stage.update.policy is stage.policy
    assert stage.update.policy.clip == 1.0
    assert stage.update.policy.reduction == "sum"
    assert stage.update.precision is torch.bfloat16


def test_frozen_parameters_stay_out_of_the_optimizer():
    """An optimizer holds no state for weights it cannot move."""
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    stage = Train("t", model=model, data=ramp(), loss=nn.MSELoss(), steps=1)
    stage.enter(Context("t"))
    held = [
        p
        for group in stage.update.optimizers[None].param_groups
        for p in group["params"]
    ]
    assert len(held) == 2
    assert all(p.requires_grad for p in held)


def test_a_fully_frozen_model_is_refused():
    """Training it would report losses and move nothing."""
    model = linear()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage = Train("t", model=model, data=ramp(), loss=nn.MSELoss(), steps=1)
    with pytest.raises(ValueError, match="none of them trainable"):
        stage.enter(Context("t"))


def test_swa_averages_only_the_tail_of_the_run():
    """Equal weighting would drag the untrained start into the mean."""
    stage = Train(
        "fit",
        model=linear(),
        data=ramp(8),
        batch_size=4,
        loss=nn.MSELoss(),
        epochs=4,
        lr=0.1,
        swa=True,
        swa_start=0.5,
    )
    Runtime(stage, hooks=(), device="cpu").run()
    assert int(stage.swa_window.averages["model"].n_averaged) == 2


def test_swa_needs_epochs_to_average_over():
    """It averages once a pass, so a step budget alone would never fire."""
    with pytest.raises(ValueError, match="swa needs epochs"):
        Train("fit", model=linear(), data=ramp(), loss=nn.MSELoss(), steps=4, swa=True)


def test_swa_start_is_a_fraction():
    """An epoch number would not port between runs of different lengths."""
    with pytest.raises(ValueError, match="fraction of the run"):
        Train(
            "fit",
            model=linear(),
            data=ramp(),
            loss=nn.MSELoss(),
            epochs=4,
            swa=True,
            swa_start=3,
        )


def test_swa_restores_the_batch_statistics_it_did_not_average():
    """Averaged weights inherit stale running stats until `update_bn` runs."""
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Linear(3, 1))
    stage = Train(
        "fit",
        model=model,
        data=ramp(16),
        batch_size=4,
        loss=nn.MSELoss(),
        epochs=4,
        lr=0.1,
        swa=True,
        swa_start=0.5,
    )
    Runtime(stage, hooks=(), device="cpu").run()
    averaged = stage.swa_window.averages["model"].module[1]
    assert not torch.equal(averaged.running_mean, torch.zeros(3))


def test_swa_rides_in_the_stage_state():
    """A resumed run keeps the average it had accumulated."""
    stage = Train(
        "fit",
        model=linear(),
        data=ramp(8),
        batch_size=4,
        loss=nn.MSELoss(),
        epochs=2,
        lr=0.1,
        swa="model",
        swa_start=0.0,
    )
    Runtime(stage, hooks=(), device="cpu").run()
    assert "swa/model" in stage.state_dict()


def test_swalr_takes_over_from_the_main_schedule():
    """The rate follows the main schedule, then whatever SWALR holds."""
    rates: list[float] = []

    class Watching:
        """Records the learning rate at each pass boundary."""

        def on(self, event):
            """Record the rate when a pass ends.

            Args:
                event: What the runtime just did.
            """
            if event.type is EventType.EPOCH_ENDED:
                rates.append(
                    round(stage.update.optimizers[None].param_groups[0]["lr"], 4)
                )
            return Signal.GO

    stage = Train(
        "fit",
        model=linear(),
        data=ramp(16),
        batch_size=8,
        loss=nn.MSELoss(),
        epochs=8,
        swa=True,
        swa_start=0.5,
        swa_lr=0.01,
        swa_anneal=2,
        optim=OptimSpec.adamw(lr=0.5).with_exponential_schedule(0.5),
    )
    Runtime(stage, hooks=(Watching(),), device="cpu").run()
    assert rates[:4] == [0.25, 0.125, 0.0625, 0.0312]
    assert rates[-2:] == [0.01, 0.01]


def test_the_main_schedule_is_retired_at_the_handover():
    """It would otherwise compete with the rate SWALR holds."""
    stage = Train(
        "fit",
        model=linear(),
        data=ramp(16),
        batch_size=8,
        loss=nn.MSELoss(),
        epochs=4,
        swa=True,
        swa_start=0.5,
        swa_lr=0.01,
        optim=OptimSpec.adamw(lr=0.5).with_exponential_schedule(0.5),
    )
    Runtime(stage, hooks=(), device="cpu").run()
    assert not stage._sweepwise_schedulers
    assert not stage.update.schedulers


def test_averaging_without_a_swa_rate_leaves_the_main_schedule_running():
    """Nothing takes the rate over, so the schedule that was there must go on."""
    rates: list[float] = []

    class Watching:
        """Records the learning rate at each pass boundary."""

        def on(self, event):
            """Record the rate when a pass ends.

            Args:
                event: What the runtime just did.
            """
            if event.type is EventType.EPOCH_ENDED:
                rates.append(
                    round(stage.update.optimizers[None].param_groups[0]["lr"], 4)
                )
            return Signal.GO

    stage = Train(
        "fit",
        model=linear(),
        data=ramp(16),
        batch_size=8,
        loss=nn.MSELoss(),
        epochs=6,
        swa=True,
        swa_start=0.5,
        optim=OptimSpec.adamw(lr=0.5).with_exponential_schedule(0.5),
    )
    Runtime(stage, hooks=(Watching(),), device="cpu").run()
    assert rates == [0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.0078]
    assert int(stage.swa_window.averages["model"].n_averaged) == 3


def test_a_swa_rate_without_averaging_is_refused():
    """It would schedule a window that never opens."""
    with pytest.raises(ValueError, match="was not asked to average"):
        Train(
            "fit",
            model=linear(),
            data=ramp(),
            loss=nn.MSELoss(),
            epochs=4,
            swa_lr=0.01,
        )


def test_the_swa_schedule_rides_in_the_stage_state():
    """A resumed run keeps where the annealing had reached."""
    stage = Train(
        "fit",
        model=linear(),
        data=ramp(16),
        batch_size=8,
        loss=nn.MSELoss(),
        epochs=4,
        swa=True,
        swa_start=0.5,
        swa_lr=0.01,
    )
    Runtime(stage, hooks=(), device="cpu").run()
    assert "swalr/None" in stage.state_dict()


def adam_steps(stage) -> int:
    """Return how many updates the optimizer has applied.

    Args:
        stage: The training stage to inspect.
    """
    optimizer = next(iter(stage.update.optimizers.values()))
    state = next(iter(optimizer.state.values()), {})
    return int(state.get("step", torch.tensor(0)))


def repeated(stage, times: int = 3):
    """Run a stage several times over, as a regimen interleaving others would.

    Args:
        stage: The stage to repeat.
        times: How many visits to make.
    """
    program = Program(
        provide={"model": linear()}, stages=[Repeat(times, Phase("cycle", [stage]))]
    )
    Runtime(program, hooks=(), device="cpu").run()
    return stage


def one_pass(name: str = "fit", **kwargs) -> Train:
    """Build a stage that takes two steps per visit.

    Args:
        name: Identifies the stage.
        kwargs: Passed on to `Train`.
    """
    return Train(
        name,
        data=ramp(8),
        batch_size=4,
        epochs=1,
        loss=nn.MSELoss(),
        optim=OptimSpec.adamw(lr=0.1),
        **kwargs,
    )


def test_a_repeated_stage_keeps_the_optimizer_it_built():
    """Interleaving an eval must not throw away the moments in between."""
    stage = repeated(one_pass())
    assert adam_steps(stage) == 6


def test_a_later_stage_brings_no_history_along():
    """Each stage builds its own, so finetuning starts from a fresh optimizer."""
    program = Program(
        provide={"model": linear()},
        stages=[one_pass(name="pretrain"), one_pass(name="refine")],
    )
    Runtime(program, hooks=(), device="cpu").run()
    assert [adam_steps(stage) for stage in program.stages] == [2, 2]


def test_a_repeated_finetune_keeps_its_optimizer_too():
    """The regimen the plan describes repeats one, so it must not reset."""
    stage = Finetune(
        "refine",
        data=ramp(8),
        batch_size=4,
        epochs=1,
        loss=nn.MSELoss(),
        optim=OptimSpec.adamw(lr=0.1),
    )
    assert repeated(stage) and adam_steps(stage) == 6


def test_a_stage_counts_the_work_of_every_entry():
    """`progress` resets on entry, so a repeated stage needs a running count."""
    stage = repeated(one_pass(), times=3)
    assert stage.progress().global_step == 2
    assert stage.total_steps == 6

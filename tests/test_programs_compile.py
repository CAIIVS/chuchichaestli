# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that a compiled model's state moves to a plain one and back."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.programs import (
    CancelAfter,
    Checkpointer,
    CheckpointStore,
    EventType,
    Load,
    Local,
    Program,
    Runner,
    Signal,
    Train,
    WeightsExport,
)
from chuchichaestli.programs.events import C3liRunCancelled
from chuchichaestli.programs.runner import C3liProgramError
from chuchichaestli.programs.topology import is_compiled
from chuchichaestli.training import OptimSpec
from chuchichaestli.utils.io import read_state

DATA = TensorDataset(torch.linspace(-1, 1, 64).reshape(16, 4), torch.ones(16, 1))


def net(compiled: bool) -> nn.Module:
    """Build a model whose initial weights are the same every time.

    Args:
        compiled: Whether to hand it out compiled.
    """
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    return torch.compile(model, backend="eager") if compiled else model


def program(compiled: bool, *more) -> Program:
    """Build a program training a model with a moving average.

    Args:
        compiled: Whether the model is compiled.
        more: Stages to run after training.
    """
    fit = Train(
        "fit",
        data=DATA,
        batch_size=4,
        epochs=2,
        loss=nn.MSELoss(),
        optim=OptimSpec.sgd(lr=0.1),
        ema=0.9,
    )
    return Program(provide={"model": net(compiled)}, stages=[fit, *more])


def plain_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a module's weights under the plain model's names.

    Args:
        module: A plain or compiled module.
    """
    return dict(Local().unwrap(module).state_dict())


@pytest.mark.parametrize(
    ("first", "then"),
    [(True, False), (False, True), (True, True)],
    ids=["compiled_to_plain", "plain_to_compiled", "compiled_to_compiled"],
)
def test_a_resume_crosses_between_compiled_and_plain(tmp_path, first, then):
    """Training compiled and resuming plain, or the reverse, ends where one run does."""
    whole = program(False)
    Runner(whole, hooks=[], device="cpu").run()
    store = tmp_path / "run"
    with pytest.raises(C3liRunCancelled):
        Runner(
            program(first),
            hooks=[Checkpointer(every=1, unit="step"), CancelAfter(steps=5)],
            store=store,
            device="cpu",
        ).run()
    carried = program(then)
    Runner(
        carried,
        hooks=[Checkpointer(every=1, unit="step")],
        store=store,
        device="cpu",
        resume="last",
    ).run()
    for name, tensor in plain_state(whole.provide["model"]).items():
        assert torch.equal(plain_state(carried.provide["model"])[name], tensor)
    averaged = carried.stages[0]._ema["model"].state_dict()
    for name, tensor in whole.stages[0]._ema["model"].state_dict().items():
        assert torch.equal(averaged[name], tensor)


def test_a_checkpoint_names_a_compiled_model_as_a_plain_one(tmp_path):
    """No `_orig_mod.` prefix reaches the file."""
    store = tmp_path / "run"
    Runner(
        program(True),
        hooks=[Checkpointer(every=1, unit="epoch")],
        store=store,
        device="cpu",
    ).run()
    saved = CheckpointStore(store).load("last").bindings["model"]
    assert set(saved) == set(net(False).state_dict())


def test_exported_weights_move_between_compiled_and_plain(tmp_path):
    """Exported from a compiled model, they load into a plain one, and back."""
    exported = tmp_path / "weights.safetensors"
    trained = program(True, WeightsExport("save", path=exported))
    Runner(trained, hooks=[], device="cpu").run()
    plain = net(False)
    plain.load_state_dict(read_state(exported))
    loaded = Program(
        provide={"model": net(True)},
        stages=[Load("load", path=exported)],
    )
    Runner(loaded, hooks=[], device="cpu").run()
    expected = plain_state(trained.provide["model"])
    for model in (plain, loaded.provide["model"]):
        for name, tensor in plain_state(model).items():
            assert torch.equal(tensor, expected[name])


EAGER = {"backend": "eager"}


class Began:
    """Keeps the payload the run began with."""

    def __init__(self):
        """Constructor."""
        self.payload: dict = {}

    def on(self, event):
        """Keep the opening event's payload.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.RUN_BEGAN:
            self.payload = dict(event.payload)
        return Signal.GO


def test_the_runner_compiles_every_module_it_holds():
    """Bound or held by a stage directly, each is compiled and trains as it would."""
    plain = program(False)
    Runner(plain, hooks=[], device="cpu").run()
    direct = Train(
        "direct",
        model=net(False),
        data=DATA,
        batch_size=4,
        epochs=1,
        loss=nn.MSELoss(),
        optim=OptimSpec.sgd(lr=0.1),
    )
    compiled = program(False, direct)
    began = Began()
    Runner(
        compiled, hooks=[began], device="cpu", compile=True, compile_kwargs=EAGER
    ).run()
    assert is_compiled(compiled.provide["model"]) and is_compiled(direct.model)
    assert began.payload["compile"] is True
    expected = plain_state(plain.provide["model"])
    for name, tensor in plain_state(compiled.provide["model"]).items():
        assert torch.equal(tensor, expected[name])


@pytest.mark.parametrize("names", [("model",), "model"], ids=["tuple", "string"])
def test_the_runner_compiles_only_the_bindings_named(names):
    """Another binding is left as it was."""
    prog = program(False)
    prog.provide["side"] = net(False)
    began = Began()
    Runner(prog, hooks=[began], device="cpu", compile=names, compile_kwargs=EAGER).run()
    assert is_compiled(prog.provide["model"])
    assert not is_compiled(prog.provide["side"])
    assert began.payload["compile"] == ("model",)


def test_compiling_an_unknown_binding_is_refused_before_the_run():
    """A typo would otherwise leave the model silently uncompiled."""
    with pytest.raises(C3liProgramError, match=r"compile names \['modle'\]"):
        Runner(program(False), hooks=[], device="cpu", compile=("modle",)).run()


def test_a_module_is_compiled_once_however_often_it_runs():
    """A second run of the same program finds it compiled already."""
    prog = program(False)
    for _ in range(2):
        Runner(prog, hooks=[], device="cpu", compile=True, compile_kwargs=EAGER).run()
    compiled = prog.provide["model"]
    assert is_compiled(compiled) and not is_compiled(compiled._orig_mod)


def test_a_run_compiled_by_the_runner_resumes_uncompiled(tmp_path):
    """The checkpoint holds plain names, so the setting may change on resume."""
    whole = program(False)
    Runner(whole, hooks=[], device="cpu").run()
    store = tmp_path / "run"
    with pytest.raises(C3liRunCancelled):
        Runner(
            program(False),
            hooks=[Checkpointer(every=1, unit="step"), CancelAfter(steps=5)],
            store=store,
            device="cpu",
            compile=True,
            compile_kwargs=EAGER,
        ).run()
    carried = program(False)
    Runner(carried, hooks=[], store=store, device="cpu", resume="last").run()
    expected = plain_state(whole.provide["model"])
    for name, tensor in plain_state(carried.provide["model"]).items():
        assert torch.equal(tensor, expected[name])


def test_holding_a_compiled_module_does_not_make_one_compiled():
    """Only DDP's wrapper is looked through, not any submodule called `module`."""
    holder = nn.Module()
    holder.module = net(True)
    assert is_compiled(holder.module) and not is_compiled(holder)

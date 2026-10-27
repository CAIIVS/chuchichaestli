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
    Load,
    Program,
    Runner,
    Train,
    WeightsExport,
)
from chuchichaestli.programs.events import C3liRunCancelled
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
    return dict(getattr(module, "_orig_mod", module).state_dict())


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

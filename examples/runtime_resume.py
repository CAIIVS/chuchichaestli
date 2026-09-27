# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: carrying a run on from a checkpoint two epochs back.

This script shows one `runtime` `Program` run twice over a single store:
- `Train` stage loop fits the model for four passes of the data,
- `Runtime(store=)` gives both runs the store the checkpoints live in,
- `Runtime(resume="last~2")` starts the second run two checkpoints back.
Each run is driven by `Runtime` including callback hooks such as
- `Checkpointer` writes a checkpoint after every pass,
- `Jsonl` records the log trace (`Jsonl.read` reads it)

The dataset provides points of `HalfMoonsDataset`, the models is a small MLP.


Run from repository root:
```
[uv run] python examples/runtime_resume.py
```
"""

# --8<-- [start:setup]
import tempfile
from pathlib import Path

import torch
from torch import nn

from chuchichaestli.data import HalfMoonsDataset
from chuchichaestli.runtime import (
    Checkpointer,
    EventType,
    Jsonl,
    Program,
    Runtime,
    Train,
)
from chuchichaestli.training import OptimSpec

STORE = Path(tempfile.mkdtemp()) / "store"
SEED = 42
torch.manual_seed(SEED)
# --8<-- [end:setup]


# --8<-- [start:program]
def build_program() -> Program:
    """Return a fresh copy of the program under test."""
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Flatten(0))
    stage = Train(
        "fit",
        data=HalfMoonsDataset(n_samples=64, noise=0.05),
        batch_size=8,
        epochs=4,
        loss=nn.MSELoss(),
        optim=OptimSpec.sgd(lr=0.1).with_exponential_schedule(0.9),
    )
    return Program(provide={"model": model}, stages=[stage])
# --8<-- [end:program]


# --8<-- [start:run]
first_round = build_program()
Runtime(
    first_round,
    seed=42,
    store=STORE,
    hooks=[Checkpointer(every=1, unit="epoch"), Jsonl("first.jsonl")],
    device="cpu",
).run()
# --8<-- [end:run]
print("Carrying on from two checkpoints back (last~2)...")
program = build_program()
# --8<-- [start:resume]
Runtime(
    program,
    seed=42,
    store=STORE,
    resume="last~2",
    hooks=[Jsonl("second_round.jsonl")],
    device="cpu",
).run()
# --8<-- [end:resume]

# --8<-- [start:compare]
expected = first_round.provide["model"].state_dict()
actual = program.provide["model"].state_dict()
identical = all(torch.equal(v, actual[k]) for k, v in expected.items())

whole_steps = Jsonl.read(STORE / "first.jsonl", only=[EventType.STEP_ENDED])
tail = Jsonl.read(STORE / "second.jsonl", only=[EventType.STEP_ENDED])
matches = tail == whole_steps[-len(tail) :]
total_matches = program.stages[0].total_steps == first_round.stages[0].total_steps

print(f"- weights identical to the bit:  {identical}")
print(f"- steps replayed after resuming: {len(tail)}")
print(f"- steps recorded (whole run):    {len(whole_steps)}")
print(f"- tail of the whole run's trace: {matches}")
print(f"- step count carried over:       {total_matches}")
print(f"\nworking files under {STORE}")
# --8<-- [end:compare]

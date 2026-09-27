# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: the smallest complete training run.

This script shows the smallest `runtime` `Program` there is:
- `Train` stage loop fits the model on a weighted sum of objective `Term`s,
- `Program` holds the stages and the bindings they share.
The program is driven by `Runtime` including callback hooks such as
- `Console` reports and times each pass,
- `Checkpointer` writes a checkpoint after every pass,
- `Jsonl` records the log trace.

The dataset provides points of `HalfMoonsDataset`, each regressed onto the
half it belongs to. Every other example builds on this one.


Run from repository root:
```
[uv run] python examples/runtime_train.py
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
    Console,
    Criterion,
    Jsonl,
    Program,
    Runtime,
    Train,
)
from chuchichaestli.training import OptimSpec, Term

STORE = Path(tempfile.mkdtemp()) / "store"
torch.manual_seed(42)

data = HalfMoonsDataset(n_samples=64, noise=0.05)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Flatten(0))
# --8<-- [end:setup]

# --8<-- [start:program]
# One stage loop: draw batches, compute the objective, step the optimizer.
# The objective is a weighted sum of named `Term`s, each reported by name.
training = Train(
    "fit",
    model="model",
    data=data,
    batch_size=8,
    epochs=4,
    objective=[
        Term("mse", Criterion(nn.MSELoss())),
        Term("mae", Criterion(nn.L1Loss()), weight=0.1),
    ],
    optim=OptimSpec.sgd(lr=0.1).with_exponential_schedule(0.9),
)

# A program holds the stages and the bindings they share
program = Program(provide={"model": model}, stages=[training])
# --8<-- [end:program]

# --8<-- [start:run]
# The runtime drives the program and reports every event to its hooks
Runtime(
    program,
    seed=42,
    hooks=[
        Console(every=1, timing=True),
        Checkpointer(every=1, unit="epoch"),
        Jsonl("trace.jsonl"),
    ],
    store=STORE,
    device="cpu",
).run()
# --8<-- [end:run]

print(f"\nTook {training.total_steps} optimization steps over {training.epochs} passes")
ckpts = sorted(p.name for p in STORE.glob("ckpt_*"))
print(f"The trace and checkpoints are stored under {STORE}")
print("- trace: trace.jsonl")
print(f"- checkpoints: {ckpts}")

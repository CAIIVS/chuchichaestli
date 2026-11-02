# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: following a run in TensorBoard, interruption and resume included.

This script trains a small upsampler on the Halfmoons dataset and logs it
with `TensorBoard`:
- `Train`, `Eval` and `Predict` run a pass at a time in `Phase.each_pass`,
- `ImageExport` plots upsampled images beside the fine ones each pass,
- `Checkpointer(monitor="mse")` writes a checkpoint each pass,
- `TensorBoard()` logs the losses, rate, gradient norm, timing, scores, the
  previews and the run's configuration, under the run's names.
The first run is stopped halfway by `CancelAfter`, as a job's time limit
would stop it, and the second resumes it from the checkpoint written as it
stopped: the curves carry on as one run.

Run from repository root, then look at the run:
```
[uv run --extra tensorboard] python examples/programs_tensorboard.py
tensorboard --logdir <output directory>
```
"""

import math
import sys
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Subset
from chuchichaestli.data import ConditionalDensityDataset, HalfMoonsDataset
from chuchichaestli.data.split import split_dataset
from chuchichaestli.metrics import MSE
from chuchichaestli.programs import (
    CancelAfter,
    Checkpointer,
    Console,
    Criterion,
    Eval,
    ImageExport,
    Phase,
    Predict,
    Program,
    Runner,
    Train,
)
from chuchichaestli.programs.events import C3liRunCancelled
from chuchichaestli.training import OptimSpec
from chuchichaestli.utils.registry import require_module

# --8<-- [start:imports]
from chuchichaestli.programs import TensorBoard
# --8<-- [end:imports]

try:
    require_module("tensorboard", "tensorboard", "this example")
except ImportError as missing:
    sys.exit(str(missing))

# --8<-- [start:setup]
STORE = Path(tempfile.mkdtemp()) / "store"
torch.manual_seed(42)

# Create an image dataset of half moons, each paired with a coarse copy
pairs = ConditionalDensityDataset(
    source=HalfMoonsDataset,
    n_images=64,
    side=32,
    points=16384,
    noise=0.05,
    return_as={"x": 0, "c": 1},
)
dataset, validation = split_dataset(pairs, (0.8, 0.2))
# --8<-- [end:setup]

# --8<-- [start:model]
EPOCHS = 16
BATCH = 4
STEPS_PER_PASS = math.ceil(len(dataset) / BATCH)

# A small convolutional upsampler
model = nn.Sequential(
    nn.Conv2d(1, 16, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(16, 16, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(16, 1, 5, padding=2),
)
# --8<-- [end:model]

# --8<-- [start:program]
# Train a pass, score it, and preview a few upsampled images
train = Train(
    "train",
    data=dataset,
    batch_size=BATCH,
    objective=Criterion(nn.MSELoss(), inputs="c", targets="x"),
    optim=OptimSpec.adamw(lr=3e-3).with_cosine_schedule(t_max=EPOCHS),
)
evaluate = Eval(
    "eval",
    data=validation,
    batch_size=BATCH,
    inputs="c",
    targets="x",
    metrics=[MSE()],
)
sample = Predict(
    "sample",
    data=Subset(validation, range(4)),
    batch_size=4,
    inputs="c",
    targets="x",
)
preview = ImageExport(
    "preview",
    path=STORE / "previews" / "upsampled_{epoch:02d}.png",
    source="sample/pairs",
    labels=("upsampled", "fine"),
)
program = Program(
    provide={"model": model},
    stages=[Phase.each_pass(EPOCHS, train, evaluate, sample, preview)],
)
# --8<-- [end:program]

# --8<-- [start:run]
# Stop halfway, as a job's time limit would, then resume where it stopped
halfway = CancelAfter(steps=STEPS_PER_PASS * EPOCHS // 2)
for resume, budget in ((None, [halfway]), ("last", [])):
    try:
        Runner(
            program,
            hooks=[
                Console(every=STEPS_PER_PASS),
                Checkpointer(every=1, unit="epoch", monitor="mse"),
                # --8<-- [start:tracker]
                TensorBoard(),
                # --8<-- [end:tracker]
                *budget,
            ],
            store=STORE,
            device="cpu",
            project="chuchichaestli-examples",
            experiment="upsample",
            run_name="tensorboard",
            resume=resume,
        ).run()
    except C3liRunCancelled as stopped:
        print(f"\n{stopped}; resuming from where it stopped\n")
# --8<-- [end:run]

print(f"\nTensorBoard files: {STORE / 'tensorboard'}")
print(f"Look at the run with: tensorboard --logdir {STORE / 'tensorboard'}")

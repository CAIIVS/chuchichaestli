# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: training a U-Net for denoising a diffusion process.

This script shows a complete `runtime` pipeline `Program` arranging
the stage loop sequence (via `Phase.each_pass`):
- `Train` stage loop fitting the model on a weighted sum of objective `Term`s,
- `Sample` publishes the diffusion model's predictions and ground truths (GTs),
- `Eval` stage scores the predictions against the GTs,
- `ImageExport` plots each published sample and its GT to its own file.
The program is driven by `Runtime` including callback hooks such as
- `Console` reports and times each pass,
- `Timer` reports walltime of the total program,
- `ProgressBar` pins the progress on the stdout line,
- `Jsonl` records the log trace, and `Jsonl.read` plays it back,
- `Checkpointer` writes checkpoints into a store.

The dataset provides images of `HalfMoonsDataset` densities. The task is
conditional: each coarse image is refined into a fine one.


Run from repository root (plotting needs `chuchichaestli[viz]`):
```
[uv run --extra viz] python examples/runtime_diffusion.py
```
"""

# --8<-- [start:setup]
import sys
import tempfile
from pathlib import Path
import torch
from torch import nn
from chuchichaestli.data import ConditionalDensityDataset, HalfMoonsDataset
from chuchichaestli.data.split import split_dataset
from chuchichaestli.diffusion.processes import DDPM
from chuchichaestli.metrics import FID, PSNR, SSIM
from chuchichaestli.models.unet import UNet
from chuchichaestli.runtime import (
    Checkpointer,
    Console,
    Eval,
    EventType,
    Every,
    Diffusion,
    FromProcess,
    ImageExport,
    Predict,
    Program,
    ProgressBar,
    Timer,
    Runtime,
    Jsonl,
    Phase,
    Train,
)
from chuchichaestli.training import OptimSpec, Term

EPOCHS = 128
STORE = Path(tempfile.mkdtemp()) / "store"
SEED = 42
torch.manual_seed(SEED)

# Create an image dataset of half moons
pairs = ConditionalDensityDataset(
    source=HalfMoonsDataset,
    n_images=64,
    side=32,
    points=16384,
    noise=0.05,
    return_as={"x": 0, "c": 1},
    seed=SEED,
)
dataset, validation = split_dataset(pairs, (0.8, 0.2), seed=SEED)
# --8<-- [end:setup]

# --8<-- [start:model]
# Build a U-Net denoising diffusion model...
model = UNet(
    dimensions=2,
    in_channels=2,
    out_channels=1,
    n_channels=16,
    down_block_types=("DownBlock", "DownBlock"),
    up_block_types=("UpBlock", "UpBlock"),
    block_out_channel_mults=(1, 2),
    res_groups=8,
    time_embedding=True,
)
# ...using a standard DDPM sampler
process = DDPM(num_timesteps=200, device="cpu")
# --8<-- [end:model]

# --8<-- [start:stages]
# Training stage using a custom objective (MSE + 0.1 x MAE)
training = Train(
    "train_denoising",
    model="model",
    data=dataset,
    batch_size=16,
    epochs=EPOCHS,
    objective=[
        Term("mse", Diffusion(process, condition="c")),
        Term("mae", Diffusion(process, condition="c", loss=nn.L1Loss()), weight=0.1),
    ],
    optim=OptimSpec.adamw(lr=1e-3).with_cosine_schedule(t_max=EPOCHS),
    ema=0.99,
)

# Samples the diffusion model from the ema weights that `Train` published,
# rather than the live ones, and publishes sample/pairs (predictions with GTs);
# `weights="model"` (default) would read the live parameters instead
sampling = Predict(
    "sample",
    model="model",
    weights="ema",
    draw=FromProcess(process),
    data=validation,
    batch_size=len(validation),
    inputs="c",
    targets="x",
)

# Runs evaluation on published (predictions with GTs)
scoring = Eval(
    "score",
    model=None,
    data="sample/pairs",
    batch_size=len(validation),
    metrics=[PSNR(), SSIM(), FID()],
)

# Plots the same published pairs, one file per image, per epoch via `{epoch}`
preview = ImageExport(
    "preview",
    path=STORE / "samples_ema_{epoch:02d}.png",
    source="sample/pairs",
    labels=("sampled", "gt"),
    limit=4,
    cmap="magma",
    normalize="shared",
)
# --8<-- [end:stages]

# --8<-- [start:run]
# Compose a pipeline program with training and validation stage...
program = Program(
    provide={"model": model},
    stages=[
        Phase.each_pass(
            EPOCHS,
            training,
            Every(32, Phase("score", [sampling, scoring, preview])),
        )
    ],
)

# ...and run it with multiple hooks
Runtime(
    program,
    seed=SEED,
    hooks=[
        Console(every=1, timing=True),
        Timer(stream=sys.stdout, depth=1),
        ProgressBar(unit="epoch"),
        Jsonl("trace.jsonl"),
        Checkpointer(every=32, unit="epoch", keep=2),
    ],
    store=STORE,
    device="cpu",
).run()
# --8<-- [end:run]

print(f"\nTook {training.total_steps} optimization steps over {EPOCHS} passes")
print(f"Sampled from the averaged weights published as {'model/ema'!r}")
print(f"The trace and checkpoints are stored under {STORE}")
ckpts = sorted(p.name for p in (STORE).glob("ckpt_*"))
steps = Jsonl.read(STORE / "trace.jsonl", only=[EventType.STEP_ENDED])
losses = [e.payload["loss"] for e in steps if "loss" in e.payload]
print(f"- trace: trace.jsonl, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
print(f"- checkpoints: {ckpts}")
print(f"- previews: {len(list(STORE.glob('samples_*.png')))} images under {STORE}")

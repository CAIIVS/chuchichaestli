# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: training a Pix2Pix GAN to refine a coarse image.

This script shows a complete `runtime` pipeline `Program` arranging
the stage loop sequence (via `Phase.each_pass`):
- `Train` stage loop fitting two models at once, a `Term` per update group,
- `Sample` publishes the generator's predictions and ground truths (GTs),
- `Eval` stage scores the predictions against the GTs,
- `ImageExport` plots each published sample and its GT to its own file.
The program is driven by `Runtime` including callback hooks such as
- `Console` reports and times each pass,
- `Timer` reports walltime of the total program,
- `ProgressBar` pins the progress on the stdout line,
- `Jsonl` records the log trace, and `Jsonl.read` plays it back,
- `Checkpointer` writes checkpoints into a store.

The generator and the discriminator are two update groups of one `Train`,
stepped in turn by `Alternating`, each with an optimizer over the parameters
its `params=` names. The discriminator is conditional: it scores the coarse
image joined to a fine one, so it cannot win on blurriness alone. The
adversarial term carries an `AdaptiveWeight` rather than a number, which
balances its gradient against the L1 term's at the layer it names.

The dataset provides images of `HalfMoonsDataset` densities. The task is
conditional: each coarse image is refined into a fine one.


Run from repository root (plotting needs `chuchichaestli[viz]`):
```
[uv run --extra viz] python examples/runtime_pix2pix.py
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
from chuchichaestli.metrics import FID, PSNR, SSIM
from chuchichaestli.models.adversarial.discriminator import PatchDiscriminator
from chuchichaestli.models.unet import UNet
from chuchichaestli.runtime import (
    Alternating,
    Checkpointer,
    Console,
    Criterion,
    DiscriminatorAdv,
    Eval,
    EventType,
    Every,
    GeneratorAdv,
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
from chuchichaestli.training import AdaptiveWeight, OptimSpec, Term

EPOCHS = 32
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
# Build a U-Net generator mapping a coarse image to a fine one...
model = UNet(
    dimensions=2,
    in_channels=1,
    out_channels=1,
    n_channels=16,
    down_block_types=("DownBlock", "DownBlock"),
    up_block_types=("UpBlock", "UpBlock"),
    block_out_channel_mults=(1, 2),
    res_groups=8,
)
# ...and a PatchGAN discriminator for adversarial scoring
disc = PatchDiscriminator(
    dimensions=2,
    in_channels=2,
    n_channels=16,
    n_hidden=2,
)
# --8<-- [end:model]

# --8<-- [start:stages]
# Training stage using the Pix2Pix objective (weighted L1 + adversarial), the
# adversarial term weighed against L1's gradient at the generator's last layer.
# Both take the same `variant` ("bce", "hinge", "least_squares", "wasserstein").
training = Train(
    "train_pix2pix",
    model="model",
    data=dataset,
    batch_size=16,
    epochs=EPOCHS,
    objective=[
        Term(
            "l1",
            Criterion(nn.L1Loss(), inputs="c", targets="x"),
            weight=10.0,
            groups=("gen",),
        ),
        # Only the discriminator half scores a real sample, so only it takes
        # targets=; the generator half scores what it produced and nothing else
        Term(
            "gen",
            GeneratorAdv(inputs="c", condition="c", variant="bce"),
            weight=AdaptiveWeight(ref="l1", layer="out_block.conv.weight"),
            groups=("gen",),
        ),
        Term(
            "disc",
            DiscriminatorAdv(inputs="c", targets="x", condition="c", variant="bce"),
            groups=("disc",),
        ),
    ],
    optim={
        "gen": OptimSpec.adam(lr=2e-4, params="model", betas=(0.5, 0.999)),
        "disc": OptimSpec.adam(lr=2e-4, params="disc", betas=(0.5, 0.999)),
    },
    update=Alternating(("disc", "gen")),
    ema=0.99,
)

# Runs the generator and publishes sample/pairs (predictions with GTs)
sampling = Predict(
    "sample",
    model="model",
    weights="model",
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
    path=STORE / "samples_{epoch:02d}.png",
    source="sample/pairs",
    labels=("generated", "gt"),
    limit=4,
    cmap="magma",
    normalize="shared",
)
# --8<-- [end:stages]

# --8<-- [start:run]
# Compose a pipeline program with training and validation stage...
program = Program(
    provide={"model": model, "disc": disc},
    stages=[
        Phase.each_pass(
            EPOCHS,
            training,
            Every(8, Phase("score", [sampling, scoring, preview])),
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
        Checkpointer(every=8, unit="epoch", keep=2),
    ],
    store=STORE,
    device="cpu",
).run()
# --8<-- [end:run]

print(f"\nTook {training.total_steps} optimization steps over {EPOCHS} passes")
print(f"Averaged weights are published as {'model/ema'!r}")
print(f"The trace and checkpoints are stored under {STORE}")
ckpts = sorted(p.name for p in (STORE).glob("ckpt_*"))
steps = Jsonl.read(STORE / "trace.jsonl", only=[EventType.STEP_ENDED])
l1 = [e.payload["gen/l1"] for e in steps if "gen/l1" in e.payload]
adversarial = [e.payload["disc/disc"] for e in steps if "disc/disc" in e.payload]
print(f"- trace: trace.jsonl, L1 {l1[0]:.3f} -> {l1[-1]:.3f}")
print(f"- discriminator: {adversarial[0]:.3f} -> {adversarial[-1]:.3f}")
print(f"- checkpoints: {ckpts}")
print(f"- previews: {len(list(STORE.glob('samples_*.png')))} images under {STORE}")

# Runtime

The [runtime][chuchichaestli.runtime] module executes what the rest of the
package builds. Training and inference are both stages in a program that
organizes them in list.

!!!note "Note"

	Nothing in `runtime` imports [hydra](https://hydra.cc/), but every
	constructor is fully keyword-constructible so a program can be written as
	a config file and instantiated from one.

## The pipelining concepts

1. A **`Program`** is a list of stages.
2. A stage is either a one-shot **`StageBlock`** or an iterative
   **`StageLoop`** — `Train`, `Eval`, `Predict`.
3. A **`Phase`** is a stage that *contains* stages, so sub-stages can nest,
   repeat, or conditionally trigger.
4. **`Runtime`** runs the program, handing each stage a **`Context`** carrying
   its position, its randomness and the shared objects.
5. Everything that happens is a runtime **`Event`** signal; **`Hook`**s watch
   events and can checkpoint, log, or stop the run.

## Minimal example

`Runtime` takes any stage, and a single stage needs no wrapper. Defaults fill
in the rest: `optim="adamw"`, `lr=1e-4`, `hooks=[Console()]` and `seed=0`.

```python
Runtime(Train(model=unet, data=train_ds, loss=nn.MSELoss(), epochs=10)).run()
```

## Denoising Diffusion Training

[`examples/runtime_diffusion.py`](https://github.com/CAIIVS/chuchichaestli/blob/main/examples/runtime_diffusion.py)
refines a coarse image into a fine one with a DDPM. It is the reference for
**conditional sampling**: the model never sees the target directly, only the
condition.

The data is paired, and `return_as` gives each half the name every stage reads
it by — `"x"` for the samples, `"c"` for the condition. Those names are what
`inputs=`, `targets=` and `condition=` refer to later.

```python
--8<-- "examples/runtime_diffusion.py:setup"
```

The process is a separate object from the model. The U-Net takes
`in_channels=2` because sample and condition arrive concatenated, and
`time_embedding=True` because a denoiser is timestep-conditioned; `DDPM` owns
the noise schedule and the reverse loop.

```python
--8<-- "examples/runtime_diffusion.py:model"
```

Four stages, each handing the next one something by name.

`Train` fits on two `Diffusion` terms sharing one process at different
weights; `Diffusion(process, condition="c")` scores a noise prediction.

`Predict` samples. `draw=FromProcess(process)` replaces the plain single noise
prediction step with the iterative reverse loop (sampling costs 200 model calls,
training one), and `weights="ema"` reads the averaged shadow weights
(`Train(ema=0.99)` publishes as `"model/ema"`) rather than the live parameters.
It publishes what it generated beside the ground truths as `"sample/pairs"`.

`Eval` scores those pairs. `model=None` because the predictions already exist
— it reads them instead of running anything — and each metric is published as
`"score/<name>"`, which a later `When` could branch on.

`ImageExport` writes the same pairs out, one file per image, numbered by the
epoch the run had reached.

```python
--8<-- "examples/runtime_diffusion.py:stages"
```

`Phase.each_pass` slots the other stages between the training passes (epochs),
and `Every(n, ...)` runs them on every nth epoch.

```python
--8<-- "examples/runtime_diffusion.py:run"
```

The hooks are reporting on the stages. `Console` prints a line each step/pass,
`Timer` measures how long each stage took and reports once at the end, and
`ProgressBar` pins a bar for the epochs to the bottom line. `Jsonl` appends
every event to a trace (`Jsonl.read` gets them back as `Event`s), and
`Checkpointer` writes the run's state every nth epoch, keeping the last `keep`.
All of them are optional, `hooks=[]` would run the same pipeline in silence.


## Adversarial Training

[`examples/runtime_gan.py`](https://github.com/CAIIVS/chuchichaestli/blob/main/examples/runtime_gan.py)
fits the same task with a Pix2Pix GAN. It is the reference for **two models
trained against each other in one stage**, and reads the same paired dataset
as above.

```python
--8<-- "examples/runtime_gan.py:setup"
```

Two models, so the program binds two names. The generator maps one channel to
one; the discriminator takes `in_channels=2` because it scores the condition
joined to a sample.

```python
--8<-- "examples/runtime_gan.py:model"
```

A GAN is one `Train` stage with two update groups. Each `Term` names the
groups it feeds, `Alternating` steps them in turn, and each optimizer owns the
parameters its `params=` names — which is how one stage trains two models
without either optimizer touching the other's weights.

```python
--8<-- "examples/runtime_gan.py:stages"
```

`Adversarial` and its two halves read the batch the way the rest of the
objectives do: `inputs=` is what the generator is given, `targets=` is what
counts as a real sample, and `condition=` is joined channel-wise to every
sample so the discriminator scores the pair.

`variant=` picks which pair of losses the two halves minimize: `"bce"` (the
default, in its non-saturating form), `"hinge"`, `"least_squares"` or
`"wasserstein"`. The two halves are separate terms, each with its own
`variant=` (so give both the same one). Note that `"wasserstein"` only estimates
a distance while its critic stays Lipschitz, which nothing here enforces:
no weight clipping, no gradient penalty.

The adversarial term carries an `AdaptiveWeight` instead of a number, the
other feature unique to this example. It balances its own gradient against the
reference term's at a named parameter, so the ratio tracks training instead of
being guessed once: over this run it moves from about 5 to 0.005 as the L1
term converges and the adversarial one grows.

```python
--8<-- "examples/runtime_gan.py:run"
```

The run is shaped like the diffusion one — same phases, same hooks — so the
program above is the only part that differs.

## Context bindings

`model=`, `data=` and friends accept the object *or* a `str` naming a
`Context` binding. That is what lets several stages share models, and what
enables configuration through frameworks like [hydra](https://hydra.cc/).

```python
program = Program(
    provide={"model": unet},
    stages=[
        Train("pretrain", data=train_ds, loss=mse, epochs=20, ema=0.9999),
        Repeat(3, Phase("cycle", [
            Finetune("refine", data=ft_ds, epochs=1, lr=1e-5),
            Eval("validate", data=val_ds, metrics=[PSNR(), SSIM()]),
        ])),
        When(lambda ctx: ctx["validate/psnr"] > 30,
             Predict("sample", data=test_ds, weights="ema", archive="out/preds.h5")),
    ],
)
```

`Eval` publishes each metric as `f"{stage}/{key}"`, which is what
`ctx["validate/psnr"]` resolves against. `Train(ema=...)` publishes its shadow
weights as `f"{binding}/ema"`, so `weights="ema"` reads them in a later stage
and restores the live parameters afterwards.


## Resuming

A run resumed from a checkpoint behaves identically to one that was never
interrupted: the same weights to the bit, the same optimizer and schedule
state, and the same trace.

```python
--8<-- "examples/runtime_resume.py:resume"
```

`resume=` takes `"last"`, `"last~N"` for N checkpoints earlier, or a path to
one. A run that *finished* records no stage state — a `Phase` saves only the
child it is in the middle of — so resuming its last checkpoint does nothing.
To carry on training from finished weights, `Load` them into a `Finetune`
stage instead, which starts an optimizer of its own.

Randomness is derived from `(seed, stage path)` rather than replayed, so the
result does not depend on where a run was interrupted, how many dataloader
workers it used, or how many processes it ran across.

!!!warning "Bitwise on CPU, and on GPU only under `backends="strict"`"

	Derived randomness guarantees identical *inputs* on resume. It cannot
	guarantee identical *outputs* on a GPU, where cuDNN autotuning and
	atomic accumulation in some backward kernels vary between runs.


## Running across several processes

`torchrun --nproc_per_node=4 train.py` needs no code change: `Runtime` picks
`Ddp` when `RANK` and `WORLD_SIZE` are set and `Local` otherwise.

Every control decision is agreed across ranks before it takes effect, so a
hook that stops the run on one rank stops it on all of them rather than
leaving the others blocked. Metrics are combined before an `Eval` publishes
them, only the main rank writes checkpoints, and a `Predict` writes one shard
per rank and joins them when the run ends.

## Archives

`Predict(archive=...)` picks its writer from the suffix. Formats that can be
appended to stream as the batches arrive; the rest are held until the file is
written.

| suffix | written by | streams |
|---|---|---|
| `.h5`, `.hdf`, `.hdf5`, `.he5` | `Hdf5Archive` | yes |
| `.npy` | `NpyArchive` | yes |
| `.safetensors` | `SafetensorsArchive` | yes |
| `.npz` | `BufferedArchive` | no |

## Exports and previews

`Exporter` is the shared half of writing something out: the binding it reads,
the path, the rank-0 guard and the atomic write. What the two concrete stages
do with the file is all that separates them.

| stage | writes | suffix picks |
|---|---|---|
| `WeightsExport` | a binding's `state_dict`, atomically | `.safetensors`, `.pt`, `.pth` |
| `ImageExport` | one file per image | `.png`, `.jpg`, `.webp`, `.tif`, `.pdf`, `.svg` |

A path with an `{epoch}`, `{step}` or `{advance}` field numbers the exports by
where the run had got to, so a stage entered every few epochs leaves its own
files rather than overwriting the last ones. An inference pass is not an
epoch: `Predict` and `Eval` loops report `trains=False`, so only a loop that
updates the model moves the epoch and step counters. `Checkpointer(unit=...)`
counts the same way. The counters start from zero on a resume, as the
checkpointer's own interval does.

```python
ImageExport(
    "preview",
    path=WORK / "samples_{epoch:02d}.png",
    source="sample/pairs",
    labels=("sampled", "gt"),
    limit=4,
    cmap="magma",
    normalize="shared",
)
```

`ImageExport` reads whatever [`save_images`][chuchichaestli.utils.visualization.images.save_images]
accepts, which includes the `"<stage>/pairs"` a `Predict` publishes: one row
per batch, predictions in one, their truths in the next. Each image becomes
its own file, named after the target's stem, its row and its place in that
row, so `preview_{epoch:02d}.png` with two labelled rows writes
`preview_08_sampled_0.png`, `preview_08_ground-truth_0.png`, and so on. The
`{epoch}` field is the exporter's; the rest of the name is `save_images`'.

Single-channel, RGB and RGBA are all written. Raw pixels go through
torchvision alone, and matplotlib takes over as soon as a colormap, a label,
a title, a vector format or a `draw` closure asks for more than pixels.

`draw` is handed the axes and the image in `[0, 1]`, and owns both: ticks,
spines, axis labels, a legend and a colorbar are all its to set, and
`ax.get_figure()` reaches the figure.

```python
def contours(ax, image):
    shown = ax.contourf(image[0].numpy(), levels=8, cmap="magma")
    ax.set_xlabel("x [px]")
    ax.get_figure().colorbar(shown, ax=ax).set_label("density")

ImageExport("preview", path="runs/preview_{epoch}.pdf", source="sample/pairs",
            draw=contours, labels=("sampled", "ground truth"))
```

Each canvas takes its aspect from the image on it, so a 3:1 image is written
3:1 rather than adrift in a square. `size` sets the inches along the longer
side — mixed aspects still share a bounding box — and `pad` the margin around
it.

Everything past `path` and `source` is handed straight to `save_images`, the
way `OptimSpec.kwargs` reaches the optimizer: `limit` chooses what is written,
`normalize` scales it, `labels` and `title` name it, `size` and `pad` shape the
canvas, and `draw` hands the drawing itself back to you.

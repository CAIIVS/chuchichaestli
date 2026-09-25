# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that the runtime drives the package's own models and processes."""

import warnings

import pytest
import torch
from torch.utils.data import TensorDataset

from chuchichaestli.data import HalfMoonsDataset
from chuchichaestli.diffusion.ddpm import DDPM
from chuchichaestli.models.unet import UNet
from chuchichaestli.runtime import (
    Checkpointer,
    Diffusion,
    EventType,
    Program,
    Runtime,
    Signal,
    Train,
)
from chuchichaestli.runtime.events import C3liRuntimeError
from chuchichaestli.training import OptimSpec

SAMPLES = 8
BATCH = 2


def images(n: int = SAMPLES, side: int = 8, per_image: int = 200) -> TensorDataset:
    """Bin the package's own point cloud into one density image per group.

    Args:
        n: Number of images to make.
        side: Width and height of each image.
        per_image: Points binned into each image.
    """
    torch.manual_seed(3)
    cloud = HalfMoonsDataset(n_samples=n * per_image, noise=0.05)
    points = torch.stack([cloud[i][0] for i in range(len(cloud))])
    points = points[torch.randperm(len(points))]
    lowest, highest = points.min(0).values, points.max(0).values
    cells = ((points - lowest) / (highest - lowest) * (side - 1)).round().long()
    counts = torch.zeros(n, side * side)
    counts.scatter_add_(
        1,
        (cells[:, 1] * side + cells[:, 0]).reshape(n, per_image),
        torch.ones(n, per_image),
    )
    counts = counts / counts.amax(1, keepdim=True)
    return TensorDataset(counts.reshape(n, 1, side, side))


def unet() -> UNet:
    """Build a model whose initial weights are the same every time."""
    torch.manual_seed(1234)
    return UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        n_channels=8,
        down_block_types=("DownBlock",),
        up_block_types=("UpBlock",),
        block_out_channel_mults=(1,),
        res_groups=4,
        time_embedding=True,
    )


def denoising(epochs: int = 2) -> Program:
    """Build a program that trains a model to undo a diffusion process.

    Args:
        epochs: Passes over the data to make.
    """
    stage = Train(
        "diffuse",
        data=images(),
        batch_size=BATCH,
        epochs=epochs,
        objective=Diffusion(DDPM(num_timesteps=50, device="cpu")),
        optim=OptimSpec.adamw(lr=1e-3),
    )
    return Program(provide={"model": unet()}, stages=[stage])


def weights(program: Program) -> dict[str, torch.Tensor]:
    """Return a copy of every trained tensor.

    Args:
        program: Program holding the model binding.
    """
    return {
        name: tensor.detach().clone()
        for name, tensor in program.provide["model"].state_dict().items()
    }


class CancelAt:
    """Stand in for a signal arriving partway through a run."""

    def __init__(self, steps: int):
        """Constructor.

        Args:
            steps: Optimizer steps to allow before cancelling.
        """
        self.steps = steps
        self.seen = 0

    def on(self, event):
        """Cancel once enough steps have gone by.

        Args:
            event: What the runtime just did.

        Raises:
            C3liRuntimeError: Once the run has taken `steps` steps.
        """
        if event.type is EventType.STEP_ENDED:
            self.seen += 1
            if self.seen >= self.steps:
                raise C3liRuntimeError("cancelled")
        return Signal.GO


def test_a_diffusion_objective_trains_a_unet():
    """The package's own model and process, driven by the runtime."""
    losses: list[float] = []

    class Watching:
        """Records the loss of each step."""

        def on(self, event):
            """Note what a step cost.

            Args:
                event: What the runtime just did.
            """
            if event.type is EventType.STEP_ENDED and "loss" in event.payload:
                losses.append(event.payload["loss"])
            return Signal.GO

    program = denoising(epochs=4)
    before = weights(program)
    Runtime(program, seed=42, hooks=[Watching()], device="cpu").run()

    assert len(losses) == 4 * (SAMPLES // BATCH)
    half = len(losses) // 2
    assert sum(losses[half:]) / half < sum(losses[:half]) / half
    assert any(
        not torch.equal(tensor, weights(program)[name])
        for name, tensor in before.items()
    )


def test_a_diffusion_run_resumes_identically(tmp_path):
    """The headline guarantee, over the package's own process."""
    store = tmp_path / "run"
    whole = denoising()
    Runtime(whole, seed=42, hooks=[], device="cpu").run()

    with pytest.raises(C3liRuntimeError):
        Runtime(
            denoising(),
            seed=42,
            store=store,
            hooks=[Checkpointer(every=1, unit="step"), CancelAt(5)],
            device="cpu",
        ).run()
    carried = denoising()
    Runtime(carried, seed=42, store=store, resume="last", hooks=[], device="cpu").run()

    expected, actual = weights(whole), weights(carried)
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        assert torch.equal(tensor, actual[name]), name


def test_a_process_with_its_own_randomness_says_it_will_not_resume():
    """The failure is silent otherwise: plausible numbers, quietly different."""
    process = DDPM(
        num_timesteps=50, device="cpu", generator=torch.Generator().manual_seed(7)
    )
    with pytest.warns(UserWarning, match="will not reproduce"):
        Diffusion(process)


def test_the_randomness_the_runtime_captures_draws_no_warning():
    """Leaving it alone is the reproducible choice, and the default."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Diffusion(DDPM(num_timesteps=50, device="cpu"))

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Objectives that read the execution context to know what to compute."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from torch.distributions import Distribution, MultivariateNormal, Normal
from torch.distributions.kl import kl_divergence
from chuchichaestli.data.batch import BatchType, input_in_batch, unpack_batch
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.traits import DiffusionLike
from chuchichaestli.metrics.lpips import LPIPSLoss
from chuchichaestli.utils.registry import require
from chuchichaestli.training.adversarial import (
    ADV_DISC_LOSSES,
    ADV_GEN_LOSSES,
    AdversarialTypes,
)
from chuchichaestli.training.objective import (
    RECONSTRUCTION_LOSSES,
    AdaptiveWeight,
    PerceptualBackboneTypes,
    ReconstructionLossTypes,
    Loss,
    Objective,
    Term,
)


__all__ = [
    "Computes",
    "Criterion",
    "CompositeObjective",
    "Output",
    "ModelCriterion",
    "Reconstruction",
    "Perceptual",
    "KL",
    "Adversarial",
    "GeneratorAdv",
    "DiscriminatorAdv",
    "Diffusion",
]


class Computes:
    """Mixin to make a criterion callable, dispatching to its `compute`."""

    def forward(self, batch: BatchType, ctx: Context) -> Loss:
        """Return the loss `compute` produces for one batch.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context, carrying the update group being applied.
        """
        return self.compute(batch, ctx)


class Criterion(Computes, nn.Module):
    """Score a model's output against a target taken from the batch.

    Attributes:
        loss: Compares the model's output to the target.
        model: Model, or the name of a binding holding one.
        inputs: Key the model's input is read from.
        targets: Key the target is read from.
    """

    def __init__(
        self,
        loss: Callable[..., torch.Tensor],
        model: nn.Module | str = "model",
        inputs: str = "x",
        targets: str = "y",
    ):
        """Constructor.

        Args:
            loss: Compares the model's output to the target.
            model: Model, or the name of a binding holding one.
            inputs: Key the model's input is read from, for mapping batches.
            targets: Key the target is read from, for mapping batches.
        """
        super().__init__()
        self.loss = loss
        self.model = model
        self.inputs = inputs
        self.targets = targets

    def __repr__(self) -> str:
        """Return a short description of the objective."""
        return f"{type(self).__name__}({type(self.loss).__name__})"

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the loss for one batch.

        Args:
            batch: A mapping, or a pair of input and target.
            ctx: Execution context for the stage.
        """
        inputs, targets = unpack_batch(
            batch, self.inputs, self.targets, reader=type(self).__name__
        )
        return Loss(self.loss(ctx.resolve(self.model)(inputs), targets))


class CompositeObjective(Computes, Objective):
    """Terms evaluated for the group being applied, then summed.

    Attributes:
        model: Binding an adaptive weight resolves its layer against.
    """

    def __init__(self, terms: Sequence[Term], model: nn.Module | str = "model"):
        """Constructor.

        Args:
            terms: What the objective sums.
            model: Model, or the name of a binding holding one, that an
                adaptive weight takes its gradients against.
        """
        super().__init__(terms)
        self.model = model

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Sum the terms contributing to the current group and step.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context, carrying the update group being applied.
        """
        terms = self.terms(ctx.group, ctx.progress.step)
        values: dict[str, torch.Tensor] = {}
        nested: dict[str, torch.Tensor] = {}
        for term in terms:
            loss = self._evaluate(term, batch, ctx)
            values[term.name] = loss.total
            nested.update({f"{term.name}/{k}": v for k, v in loss.parts.items()})
        combined = self.combine(values, self._weights(terms, values, ctx))
        if not nested:
            return combined
        return Loss(total=combined.total, parts={**combined.parts, **nested})

    def _evaluate(self, term: Term, batch: BatchType, ctx: Context) -> Loss:
        """Return what one term evaluates to.

        Args:
            term: The term to evaluate.
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.

        Raises:
            TypeError: If the term's objective cannot compute a loss.
        """
        compute = getattr(term.criterion, "compute", None)
        if compute is None:
            raise TypeError(
                f"Term {term.name!r} holds a {type(term.criterion).__name__}, "
                "which has no compute(batch, ctx)."
            )
        return compute(batch, ctx)

    def _weights(
        self,
        terms: Sequence[Term],
        values: Mapping[str, torch.Tensor],
        ctx: Context,
    ) -> dict[str, torch.Tensor]:
        """Compute the weight of every adaptively weighted term.

        Args:
            terms: The terms contributing to this step.
            values: What each of them evaluated to.
            ctx: Execution context for the stage.

        Raises:
            ValueError: If a term is balanced against one that is not
                contributing to the same group and step.
        """
        weights: dict[str, torch.Tensor] = {}
        model: nn.Module | None = None
        for term in terms:
            if not isinstance(term.weight, AdaptiveWeight):
                continue
            adaptive = term.weight
            if adaptive.ref not in values:
                raise ValueError(
                    f"Term {term.name!r} is balanced against {adaptive.ref!r}, "
                    f"which is not contributing here; {sorted(values)} are."
                )
            if model is None:
                model = ctx.resolve(self.model)
            weights[term.name] = adaptive(
                values[adaptive.ref],
                values[term.name],
                model.get_parameter(adaptive.layer),
            )
        return weights


@dataclass(frozen=True, slots=True)
class Output:
    """What a model produced, in the one shape every term reads.

    A model returning its posterior beside its output is unpacked on
    construction, so `Output(model(x))` reads the same for either shape.

    Attributes:
        tensor: What the model returned, or the first of what it returned.
        posterior: The distribution it returned beside that, if any.
    """

    tensor: torch.Tensor | tuple[Any, ...]
    posterior: Distribution | None = None

    @property
    def prior(self) -> Distribution | None:
        """Return the normal prior this output's posterior is measured against."""
        if self.posterior is None:
            return None
        mean = torch.zeros_like(self.posterior.mean)
        if isinstance(self.posterior, MultivariateNormal):
            return MultivariateNormal(
                mean, scale_tril=torch.diag_embed(torch.ones_like(mean))
            )
        return Normal(mean, torch.ones_like(mean))

    def __post_init__(self) -> None:
        """Split a pair into the output and the distribution beside it."""
        if not isinstance(self.tensor, tuple):
            return
        returned = self.tensor
        beside = returned[1] if len(returned) > 1 else None
        object.__setattr__(self, "tensor", returned[0])
        object.__setattr__(
            self, "posterior", beside if isinstance(beside, Distribution) else None
        )


class ModelCriterion(Computes, nn.Module):
    """A criterion computed from what a model produces for a batch.

    Attributes:
        model: Model, or the name of a binding holding one.
        inputs: Key the model's input is read from.
        cache: Name the shared forward is cached under.
    """

    def __init__(
        self,
        model: nn.Module | str = "model",
        inputs: str = "x",
        cache: str = "output",
    ):
        """Constructor.

        Args:
            model: Model, or the name of a binding holding one.
            inputs: Key the model's input is read from, for mapping batches.
            cache: Name the shared forward is cached under.
        """
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.cache = cache

    def run_model(self, batch: BatchType, ctx: Context) -> tuple[Output, torch.Tensor]:
        """Return what the model produced and what it was given.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        x = input_in_batch(batch, self.inputs)
        y = ctx.cache(self.cache, lambda: Output(ctx.resolve(self.model)(x)))
        return y, x


class Reconstruction(ModelCriterion):
    """Scores a model's output against the input it was given."""

    def __init__(
        self,
        loss: ReconstructionLossTypes | Callable[..., torch.Tensor] = "l1",
        *,
        model: nn.Module | str = "model",
        inputs: str = "x",
        cache: str = "output",
        **settings: Any,
    ):
        """Constructor.

        Args:
            loss: Compares the output to the input, either named or given
                outright. Anything comparing two tensors works, including
                `SSIMLoss` and `WaveletLoss`.
            model: Model, or the name of a binding holding one.
            inputs: Key the model's input is read from, for mapping batches.
            cache: Name the shared forward is cached under.
            settings: Passed to a named loss. `"huber"` and `"smooth_l1"`
                need their transition point set for data in `[0, 1]`, where
                the default of `1.0` leaves them quadratic throughout.

        Raises:
            ValueError: If a name matches no known loss, or settings were
                given for a loss that was handed over already built.
        """
        super().__init__(model=model, inputs=inputs, cache=cache)
        if isinstance(loss, str):
            loss = require(loss, RECONSTRUCTION_LOSSES, "reconstruction loss")(
                **settings
            )
        elif settings:
            raise ValueError(
                f"{sorted(settings)} configure a named loss, but "
                f"{type(loss).__name__} was given already built."
            )
        self.loss = loss

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the reconstruction loss for one batch.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        y, x = self.run_model(batch, ctx)
        return Loss(self.loss(y.tensor, x))


class Perceptual(ModelCriterion):
    """Scores a model's output against its input through a feature network."""

    def __init__(
        self,
        loss: PerceptualBackboneTypes | nn.Module = "vgg16",
        *,
        model: nn.Module | str = "model",
        inputs: str = "x",
        cache: str = "output",
        **settings: Any,
    ):
        """Constructor.

        Args:
            loss: Compares the output to the input in feature space, either a
                backbone `LPIPSLoss` is built around or a loss given outright.
            model: Model, or the name of a binding holding one.
            inputs: Key the model's input is read from, for mapping batches.
            cache: Name the shared forward is cached under.
            settings: Passed to `LPIPSLoss` alongside a named backbone.

        Raises:
            ValueError: If settings were given for a loss that was handed over
                already built.
        """
        super().__init__(model=model, inputs=inputs, cache=cache)
        if isinstance(loss, str):
            loss = LPIPSLoss(loss, **settings)
        elif settings:
            raise ValueError(
                f"{sorted(settings)} configure a named backbone, but "
                f"{type(loss).__name__} was given already built."
            )
        self.loss = loss

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the perceptual loss for one batch.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        y, x = self.run_model(batch, ctx)
        return Loss(self.loss(y.tensor, x))


class KL(ModelCriterion):
    """Pulls a model's posterior towards a standard normal."""

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the divergence for one batch.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.

        Raises:
            TypeError: If the model emitted no distribution to diverge from.
        """
        y, _ = self.run_model(batch, ctx)
        posterior = y.posterior
        if posterior is None:
            raise TypeError(
                f"{type(self).__name__} needs a model returning its posterior "
                "beside its output, as a VAE does."
            )
        return Loss(kl_divergence(posterior, y.prior).mean())


class Adversarial(ModelCriterion):
    """A term pitting a discriminator against what a model produces.

    Attributes:
        discriminator: Discriminator, or the name of a binding holding one.
        variant: Which pair of adversarial losses to use.
        generator_loss: What the generator minimises, looked up from `variant`.
        discriminator_loss: What the discriminator minimises.
    """

    def __init__(
        self,
        discriminator: nn.Module | str = "disc",
        variant: AdversarialTypes = "bce",
        **kwargs: Any,
    ):
        """Constructor.

        Args:
            discriminator: Discriminator, or the name of a binding holding one.
            variant: Which pair of adversarial losses to use. `"wasserstein"`
                only estimates a distance while its critic stays Lipschitz,
                which nothing here enforces.
            kwargs: Passed to `ModelCriterion`.

        Raises:
            ValueError: If the variant names no known pair of losses.
        """
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.variant = variant
        self.generator_loss = require(variant, ADV_GEN_LOSSES, "adversarial variant")
        self.discriminator_loss = require(
            variant, ADV_DISC_LOSSES, "adversarial variant"
        )

    def run_discriminator(self, sample: torch.Tensor, ctx: Context) -> torch.Tensor:
        """Return the scores the discriminator gives a sample, as logits.

        Args:
            sample: What to score.
            ctx: Execution context for the stage.
        """
        return ctx.resolve(self.discriminator)(sample)


class GeneratorAdv(Adversarial):
    """Pushes the model to produce samples the discriminator calls real."""

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the non-saturating generator loss for one batch.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        y, _ = self.run_model(batch, ctx)
        fake = self.run_discriminator(y.tensor, ctx)
        return Loss(self.generator_loss(fake))


class DiscriminatorAdv(Adversarial):
    """Teaches the discriminator to tell the model's output from the input."""

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the discriminator loss for one batch.

        The model's output is detached, which keeps it out of this graph and
        so avoids the unused-parameter error a distributed backward raises.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        y, x = self.run_model(batch, ctx)
        real = self.run_discriminator(x, ctx)
        fake = self.run_discriminator(y.tensor.detach(), ctx)
        total = self.discriminator_loss(real, fake)
        return Loss(total, {"real": real.mean().detach(), "fake": fake.mean().detach()})


class Diffusion(ModelCriterion):
    """Scores a model's noise prediction against the noise that was added.

    Only the noise parameterisation is covered: the model is run on a noised
    sample and its output compared to the noise that made it.

    Attributes:
        process: Supplies the forward noising step.
        loss: Compares the prediction to the noise.
    """

    def __init__(
        self,
        process: DiffusionLike,
        loss: Callable[..., torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        """Constructor.

        Args:
            process: Diffusion process supplying `noise_step`.
            loss: Compares the prediction to the noise; MSE by default.
            kwargs: Passed to `ModelCriterion`.
        """
        super().__init__(**kwargs)
        self.process = process
        self.loss = nn.MSELoss() if loss is None else loss

    def noise(self, batch: BatchType, ctx: Context) -> tuple[Any, Any, Any]:
        """Return a noised sample, the noise that made it, and the timesteps.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        x = input_in_batch(batch, self.inputs)
        # cache so terms sharing a step see the same draw
        return ctx.cache(f"{self.cache}/noise", lambda: self.process.noise_step(x))

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the denoising loss for one batch.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.
        """
        sampled, noise, timesteps = self.noise(batch, ctx)
        y = ctx.cache(
            self.cache, lambda: Output(ctx.resolve(self.model)(sampled, timesteps))
        )
        return Loss(self.loss(y.tensor, noise))

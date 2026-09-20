# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the objective terms shipped with the runtime."""

import pytest
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset

from chuchichaestli.runtime import (
    KL,
    Diffusion,
    Adversarial,
    Alternating,
    CompositeObjective,
    Context,
    DiscriminatorAdv,
    GeneratorAdv,
    Perceptual,
    Program,
    Reconstruction,
    Runtime,
    Train,
    Output,
)
from chuchichaestli.runtime.traits import Objective
from chuchichaestli.training import (
    ADV_DISC_LOSSES,
    ADV_GEN_LOSSES,
    Loss,
    OptimSpec,
    Term,
)


class Counting(nn.Module):
    """A model that records how often it was run."""

    def __init__(self, posterior: bool = False):
        """Constructor.

        Args:
            posterior: Whether to emit a distribution beside its output.
        """
        super().__init__()
        self.layer = nn.Linear(3, 3, bias=False)
        self.posterior = posterior
        self.calls = 0

    def forward(self, x: torch.Tensor):
        """Run the model.

        Args:
            x: Input batch.
        """
        self.calls += 1
        out = self.layer(x)
        if not self.posterior:
            return out
        scale = torch.ones_like(out) * 0.5
        return out, MultivariateNormal(out, scale_tril=torch.diag_embed(scale))


def a_context(**bindings) -> Context:
    """Build a root context holding the given bindings.

    Args:
        bindings: Artifacts to bind.
    """
    return Context("t", bindings=bindings)


def test_reconstruction_scores_the_output_against_the_input():
    """A perfect reconstruction costs nothing."""
    model = nn.Identity()
    x = torch.ones(4, 3)
    loss = Reconstruction(nn.L1Loss()).compute((x,), a_context(model=model))
    assert float(loss.total) == 0.0


def test_perceptual_runs_through_a_feature_network():
    """The term delegates to whatever compares in feature space."""
    seen: list[tuple[int, ...]] = []

    class Stub(nn.Module):
        """Stands in for LPIPS without the weights."""

        def forward(self, a, b):
            """Record the shapes and return a scalar.

            Args:
                a: The output.
                b: What it is compared against.
            """
            seen.append(tuple(a.shape))
            return (a - b).abs().mean()

    x = torch.ones(2, 3)
    Perceptual(Stub()).compute((x,), a_context(model=nn.Identity()))
    assert seen == [(2, 3)]


def test_kl_needs_a_posterior():
    """A model returning only a tensor has nothing to diverge from."""
    with pytest.raises(TypeError, match="needs a model returning its posterior"):
        KL().compute((torch.ones(2, 3),), a_context(model=nn.Identity()))


def test_kl_falls_to_zero_for_a_standard_posterior():
    """A posterior that is already standard costs nothing."""

    class Standard(nn.Module):
        """Emits exactly the distribution KL measures against."""

        def forward(self, x):
            """Return the input and a standard normal.

            Args:
                x: Input batch.
            """
            mean = torch.zeros_like(x)
            return x, MultivariateNormal(
                mean, scale_tril=torch.diag_embed(torch.ones_like(mean))
            )

    loss = KL().compute((torch.ones(2, 3),), a_context(model=Standard()))
    assert float(loss.total) == pytest.approx(0.0, abs=1e-6)


def test_the_shared_forward_runs_once_per_group_per_step():
    """Terms sharing a model pay for it once, which is why the cache exists."""
    model = Counting()
    objective = CompositeObjective(
        [
            Term("rec", Reconstruction(nn.L1Loss())),
            Term("perc", Reconstruction(nn.MSELoss())),
        ]
    )
    ctx = a_context(model=model)
    objective.compute((torch.ones(4, 3),), ctx.at_group("gen"))
    assert model.calls == 1
    objective.compute((torch.ones(4, 3),), ctx.at_group("disc"))
    assert model.calls == 2


def test_the_discriminator_group_sees_a_detached_output():
    """Detaching keeps the generator out of the discriminator's graph."""
    model = Counting()
    term = DiscriminatorAdv()
    ctx = a_context(model=model, disc=nn.Linear(3, 1))
    loss = term.compute((torch.ones(4, 3),), ctx.at_group("disc"))
    loss.total.backward()
    assert model.layer.weight.grad is None


def test_the_generator_group_carries_gradient_into_the_model():
    """The generator's loss must reach the model it is training."""
    model = Counting()
    term = GeneratorAdv()
    ctx = a_context(model=model, disc=nn.Linear(3, 1))
    term.compute((torch.ones(4, 3),), ctx.at_group("gen")).total.backward()
    assert model.layer.weight.grad is not None


def test_the_discriminator_reports_both_halves():
    """Its parts say how it did on real samples and on produced ones."""
    ctx = a_context(model=Counting(), disc=nn.Linear(3, 1))
    loss = DiscriminatorAdv().compute((torch.ones(4, 3),), ctx.at_group("disc"))
    assert sorted(loss.parts) == ["fake", "real"]


def test_an_output_reads_either_shape_a_model_returns():
    """A tensor, or a tensor beside a distribution, normalise to one record."""
    x = torch.ones(2, 3)
    plain = Output(x)
    assert plain.tensor is x and plain.posterior is None

    posterior = MultivariateNormal(x, scale_tril=torch.diag_embed(torch.ones_like(x)))
    pair = Output((x, posterior))
    assert pair.tensor is x and pair.posterior is posterior


def test_an_output_ignores_a_second_item_that_is_not_a_distribution():
    """Only a distribution counts as a posterior."""
    x = torch.ones(2, 3)
    assert Output((x, "metadata")).posterior is None


def test_each_group_touches_only_its_own_parameters():
    """On a step the generator sits out, its weights do not move."""
    torch.manual_seed(0)
    gen, disc = nn.Linear(3, 3, bias=False), nn.Linear(3, 1, bias=False)
    data = TensorDataset(torch.randn(8, 3), torch.zeros(8, 1))
    stage = Train(
        "gan",
        data=data,
        batch_size=4,
        steps=2,
        objective=[
            Term("gadv", GeneratorAdv(), groups=("gen",)),
            Term("dadv", DiscriminatorAdv(), groups=("disc",)),
        ],
        optim={
            "disc": OptimSpec.adam(lr=0.05, params="disc"),
            "gen": OptimSpec.adam(lr=0.05, params="model"),
        },
        update=Alternating(("disc", "gen"), {"disc": 5}),
    )
    ctx = Context("t", bindings={"model": gen, "disc": disc})
    stage.enter(ctx)

    stage.execute(ctx)
    assert stage.update.groups(0) == ("disc", "gen")
    after_first = gen.weight.detach().clone()
    disc_first = disc.weight.detach().clone()

    stage.execute(ctx)
    assert stage.update.groups(1) == ("disc",)
    assert torch.equal(gen.weight, after_first)
    assert not torch.equal(disc.weight, disc_first)


def test_a_two_group_stage_trains_both(tmp_path):
    """Both models move, and each keeps its own optimizer state.

    Args:
        tmp_path: Unused; keeps the signature uniform.
    """
    torch.manual_seed(0)
    gen, disc = nn.Linear(3, 3, bias=False), nn.Linear(3, 1, bias=False)
    data = TensorDataset(torch.randn(8, 3), torch.zeros(8, 1))
    g0, d0 = gen.weight.detach().clone(), disc.weight.detach().clone()
    stage = Train(
        "gan",
        data=data,
        batch_size=4,
        epochs=1,
        objective=[
            Term("rec", Reconstruction(nn.L1Loss()), groups=("gen",)),
            Term("gadv", GeneratorAdv(), groups=("gen",)),
            Term("dadv", DiscriminatorAdv(), groups=("disc",)),
        ],
        optim={
            "disc": OptimSpec.adam(lr=0.05, params="disc"),
            "gen": OptimSpec.adam(lr=0.05, params="model"),
        },
        update=Alternating(("disc", "gen")),
    )
    program = Program(provide={"model": gen, "disc": disc}, stages=[stage])
    Runtime(program, hooks=(), device="cpu").run()
    assert not torch.equal(gen.weight, g0)
    assert not torch.equal(disc.weight, d0)
    assert sorted(k for k in stage.state_dict() if k.startswith("optim")) == [
        "optim/disc",
        "optim/gen",
    ]


def test_a_criterion_is_callable_like_any_module():
    """`forward` dispatches to `compute`, so these behave as modules do."""
    term = Reconstruction(nn.L1Loss())
    ctx = a_context(model=nn.Identity())
    batch = (torch.ones(2, 3),)
    assert float(term(batch, ctx).total) == float(term.compute(batch, ctx).total)


def test_the_protocol_stays_free_of_nn_module():
    """An objective need not be a module; `compute` is the whole contract."""

    class Bare:
        """A user's own objective, inheriting nothing."""

        def compute(self, batch, ctx):
            """Return a fixed loss.

            Args:
                batch: Ignored.
                ctx: Ignored.
            """
            return Loss(torch.zeros(()))

    assert isinstance(Bare(), Objective)


@pytest.mark.parametrize(("name", "expected"), [("l1", nn.L1Loss), ("l2", nn.MSELoss)])
def test_reconstruction_names_its_loss(name, expected):
    """A name picks the comparison, so a config file can choose it.

    Args:
        name: The loss to ask for.
        expected: The class it should resolve to.
    """
    assert isinstance(Reconstruction(name).loss, expected)


def test_reconstruction_still_takes_a_loss_outright():
    """Anything comparing two tensors works, named or not."""
    assert isinstance(Reconstruction(nn.SmoothL1Loss()).loss, nn.SmoothL1Loss)


def test_an_unknown_reconstruction_loss_lists_the_known_ones():
    """A typo says what was available."""
    with pytest.raises(ValueError, match="Unsupported reconstruction loss"):
        Reconstruction("cauchy")


def test_the_named_losses_differ_in_what_they_measure():
    """l1 and l2 disagree on the same batch, which is the point of choosing."""
    ctx = a_context(model=nn.Linear(2, 2, bias=False))
    batch = (torch.tensor([[2.0, 0.0]]),)
    first = float(Reconstruction("l1").compute(batch, ctx).total)
    second = float(Reconstruction("l2").compute(batch, ctx.at_group("other")).total)
    assert first != second


def test_a_named_loss_takes_its_transition_point():
    """Huber is only robust once its knob is set for the data's scale."""
    assert Reconstruction("huber", delta=0.1).loss.delta == 0.1
    assert Reconstruction("smooth_l1", beta=0.05).loss.beta == 0.05


def test_the_default_transition_point_is_merely_quadratic():
    """At delta=1 on [0, 1] data, huber is half of l2 — hence the knob."""
    a, b = torch.tensor([[0.1, 0.3, 0.05]]), torch.zeros(1, 3)
    default = Reconstruction("huber").loss(a, b)
    assert torch.allclose(default, 0.5 * nn.MSELoss()(a, b))
    assert not torch.allclose(Reconstruction("huber", delta=0.1).loss(a, b), default)


def test_settings_are_refused_for_a_loss_given_outright():
    """They would be silently dropped, since there is nothing to build."""
    with pytest.raises(ValueError, match="given already built"):
        Reconstruction(nn.L1Loss(), delta=0.1)


def test_the_model_binding_is_still_selectable():
    """Keyword arguments for the criterion are not confused with the loss."""
    assert Reconstruction("l1", model="gen").model == "gen"


@pytest.mark.parametrize("backbone", ["squeezenet", "resnet18"])
def test_perceptual_names_its_backbone(backbone):
    """A name picks the feature network, so a config file can choose it.

    Args:
        backbone: The feature network to ask for.
    """
    from chuchichaestli.metrics import LPIPSLoss

    term = Perceptual(backbone)
    assert isinstance(term.loss, LPIPSLoss)


def test_perceptual_still_takes_a_loss_outright():
    """Any module comparing two tensors works, named or not."""
    assert isinstance(Perceptual(nn.L1Loss()).loss, nn.L1Loss)


def test_perceptual_settings_are_refused_for_a_built_loss():
    """They would be silently dropped, since there is nothing to build."""
    with pytest.raises(ValueError, match="given already built"):
        Perceptual(nn.L1Loss(), finetune=True)


def test_an_output_offers_the_prior_its_posterior_is_measured_against():
    """The prior matches the posterior's class, since KL dispatches on both."""
    x = torch.ones(2, 3)
    posterior = MultivariateNormal(x, scale_tril=torch.diag_embed(torch.ones_like(x)))
    prior = Output((x, posterior)).prior
    assert isinstance(prior, MultivariateNormal)
    assert torch.equal(prior.mean, torch.zeros_like(x))


def test_an_output_without_a_posterior_has_no_prior():
    """There is nothing to measure against."""
    assert Output(torch.ones(2, 3)).prior is None


@pytest.mark.parametrize("variant", ["bce", "hinge", "least_squares", "wasserstein"])
def test_every_adversarial_variant_trains_both_sides(variant):
    """Each variant gives a gradient to the model and to the discriminator.

    Args:
        variant: The pair of adversarial losses under test.
    """
    model, disc = Counting(), nn.Linear(3, 1)
    ctx = a_context(model=model, disc=disc)
    batch = (torch.ones(4, 3),)

    GeneratorAdv(variant=variant).compute(batch, ctx.at_group("gen")).total.backward()
    assert model.layer.weight.grad is not None

    fresh = a_context(model=Counting(), disc=disc)
    loss = DiscriminatorAdv(variant=variant).compute(batch, fresh.at_group("disc"))
    loss.total.backward()
    assert disc.weight.grad is not None


def test_hinge_and_wasserstein_share_a_generator_loss():
    """They differ on the discriminator side only."""
    fake = torch.tensor([-1.0, 0.5])
    assert ADV_GEN_LOSSES["hinge"](fake) == ADV_GEN_LOSSES["wasserstein"](fake)
    real = torch.tensor([2.0, 1.5])
    assert ADV_DISC_LOSSES["hinge"](real, fake) != ADV_DISC_LOSSES["wasserstein"](
        real, fake
    )


def test_the_variants_disagree_on_the_same_scores():
    """Choosing one is a real choice, not a relabelling."""
    real, fake = torch.tensor([2.0, 1.5]), torch.tensor([-1.0, 0.5])
    values = {n: float(f(real, fake)) for n, f in ADV_DISC_LOSSES.items()}
    assert len(set(values.values())) == len(values)


def test_an_unknown_variant_lists_the_known_ones():
    """A typo says what was available."""
    with pytest.raises(ValueError, match="Unsupported adversarial variant"):
        Adversarial(variant="relativistic")


def test_an_adversarial_term_holds_its_loss_pair():
    """The variant is resolved once, at construction, not per step."""
    term = Adversarial(variant="hinge")
    assert term.generator_loss is ADV_GEN_LOSSES["hinge"]
    assert term.discriminator_loss is ADV_DISC_LOSSES["hinge"]


def test_both_halves_of_every_variant_exist():
    """The two maps are keyed alike, so a lookup in one is safe for the other."""
    assert sorted(ADV_GEN_LOSSES) == sorted(ADV_DISC_LOSSES)


class Noising:
    """A stand-in diffusion process with a known, recorded draw."""

    def __init__(self):
        """Constructor."""
        self.calls = 0

    def noise_step(self, x, *args, **kwargs):
        """Return a noised sample, its noise and its timesteps.

        Args:
            x: Clean samples to noise.
            args: Unused.
            kwargs: Unused.
        """
        self.calls += 1
        noise = torch.full_like(x, 0.25)
        return x + noise, noise, torch.zeros(len(x), dtype=torch.long)


def test_diffusion_scores_the_prediction_against_the_noise():
    """A model returning the noise exactly costs nothing."""

    class Perfect(nn.Module):
        """Predicts the noise it was given."""

        def forward(self, sampled, timesteps):
            """Return the noise.

            Args:
                sampled: The noised sample.
                timesteps: Unused.
            """
            return torch.full_like(sampled, 0.25)

    term = Diffusion(Noising(), model=Perfect())
    loss = term.compute((torch.ones(4, 3),), a_context())
    assert float(loss.total) == pytest.approx(0.0)


def test_diffusion_shares_its_draw_within_a_step():
    """Two terms in one step see the same noise, not two draws."""
    process = Noising()
    ctx = a_context(model=nn.Identity())
    first = Diffusion(process, model=nn.Identity())
    second = Diffusion(process, model=nn.Identity())
    batch = (torch.ones(4, 3),)
    first.noise(batch, ctx)
    second.noise(batch, ctx)
    assert process.calls == 1


def test_diffusion_inherits_the_model_criterion_configuration():
    """It shares `model`, `inputs` and `cache` rather than redeclaring them."""
    from chuchichaestli.runtime.objective import ModelCriterion

    term = Diffusion(Noising(), model="unet", inputs="image")
    assert isinstance(term, ModelCriterion)
    assert (term.model, term.inputs) == ("unet", "image")

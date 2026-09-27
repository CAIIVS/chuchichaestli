# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the per-entry execution context."""

import pytest
import torch

from chuchichaestli.runtime.context import Context, C3liContextError
from chuchichaestli.runtime.events import EventType, Progress, Signal


def test_child_paths_are_built_from_index_and_name():
    """The path is the one source of truth for keys, streams and labels."""
    root = Context("program", seed=1)
    child = root.child(2, "cycle").child(0, "refine")
    assert child.path == "program/2:cycle/0:refine"


def test_bindings_resolve_outwards():
    """A child sees what its ancestors bound."""
    root = Context("program", seed=1, bindings={"model": "M"})
    assert root.child(0, "a")["model"] == "M"
    assert "model" in root.child(0, "a")
    assert root.child(0, "a").get("absent", "fallback") == "fallback"


def test_bind_is_local_and_publish_reaches_later_siblings():
    """Children inherit what a stage binds; results must outlive it."""
    root = Context("program", seed=1)
    first, second = root.child(0, "a"), root.child(1, "b")
    first.bind("scratch", 1)
    first.publish("a/score", 31.4)
    assert "scratch" not in second
    assert second["a/score"] == 31.4
    assert first.child(0, "inner")["scratch"] == 1


def test_unbound_names_what_is_available():
    """The error has to be actionable, not just a KeyError."""
    ctx = Context("program", seed=1, bindings={"model": 1})
    with pytest.raises(C3liContextError, match="model"):
        ctx["nope"]


def test_resolve_accepts_objects_or_names():
    """This is what stops a config framework instantiating a second copy."""
    ctx = Context("program", seed=1, bindings={"model": "M"})
    sentinel = object()
    assert ctx.resolve("model") == "M"
    assert ctx.resolve(sentinel) is sentinel


def test_rng_depends_on_position_not_on_call_order():
    """Two contexts at the same position give the same stream."""
    root = Context("program", seed=9)
    a, b = root.child(0, "train"), root.child(1, "eval")
    assert a.seed_for("data") != b.seed_for("data")
    assert a.seed_for("data") == root.child(0, "train").seed_for("data")
    assert a.seed_for("data") != a.seed_for("noise")


def test_cache_computes_once_per_group():
    """Composed terms share one forward; different graphs do not."""
    calls: list[str | None] = []

    def make(ctx):
        calls.append(ctx.group)
        return f"x-{ctx.group}"

    ctx = Context("program", seed=1)
    gen, disc = ctx.at_group("gen"), ctx.at_group("disc")
    assert gen.cache("recon", lambda: make(gen)) == "x-gen"
    assert gen.cache("recon", lambda: make(gen)) == "x-gen"
    assert disc.cache("recon", lambda: make(disc)) == "x-disc"
    assert calls == ["gen", "disc"]


def test_clear_cache_is_shared_by_group_views():
    """One clear at the end of a step drops every group's intermediates."""
    ctx = Context("program", seed=1)
    gen = ctx.at_group("gen")
    gen.cache("k", lambda: 1)
    ctx.clear_cache()
    assert gen.cache("k", lambda: 2) == 2


def test_child_gets_a_fresh_cache():
    """A child stage is a different scope, not a continuation."""
    ctx = Context("program", seed=1)
    ctx.cache("k", lambda: 1)
    assert ctx.child(0, "a").cache("k", lambda: 2) == 2


def test_emit_builds_the_event_from_path_and_progress():
    """A stage says what happened; the context knows where and when."""
    seen = []

    def dispatch(event):
        seen.append(event)
        return Signal.GO

    ctx = Context("program", seed=1, dispatch=dispatch)
    ctx.progress = Progress(step=4)
    assert ctx.emit(EventType.STEP_ENDED, loss=0.25) is Signal.GO
    assert seen[0].path == "program"
    assert seen[0].progress.step == 4
    assert seen[0].payload == {"loss": 0.25}


def test_emit_without_a_dispatcher_is_a_no_op():
    """A context built outside a run is still usable."""
    assert Context("p", seed=1).emit(EventType.RUN_BEGAN) is Signal.GO


def test_stateful_lists_only_the_bindings_that_carry_state():
    """A checkpoint wants the models and optimizers, not the scalars."""
    model = torch.nn.Linear(2, 2)
    ctx = Context("program")
    ctx.bind("model", model)
    ctx.bind("optim", torch.optim.Adam(model.parameters()))
    ctx.bind("threshold", 0.5)
    ctx.bind("name", "unet")
    assert sorted(ctx.stateful()) == ["model", "optim"]
    assert ctx.stateful()["model"] is model


def test_stateful_sees_what_an_enclosing_context_bound():
    """Bindings resolve outwards, and so does this."""
    root = Context("program")
    root.bind("model", torch.nn.Linear(2, 2))
    child = root.child(0, "train")
    child.bind("optim", torch.optim.Adam(root["model"].parameters()))
    assert sorted(child.stateful()) == ["model", "optim"]
    assert sorted(root.stateful()) == ["model"]

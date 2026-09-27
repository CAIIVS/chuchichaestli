# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the runtime driver: pre-flight, control flow and placement."""

import socket

import pytest
import torch
from torch import nn

from chuchichaestli.runtime.events import (
    C3liRuntimeError,
    Event,
    EventType,
    Progress,
    Signal,
)
from chuchichaestli.runtime.hooks import Console, Jsonl
from chuchichaestli.runtime.runtime import (
    C3liProgramError,
    Runtime,
    apply_backends_settings,
)
from chuchichaestli.runtime.stages import Call, Program, Repeat
from chuchichaestli.runtime.topology import Ddp, Local, auto_topology, default_device


class Record:
    """A hook keeping every event it saw."""

    def __init__(self, verdicts: dict[tuple[EventType, str], Signal] | None = None):
        """Constructor.

        Args:
            verdicts: Signal to return for a given event type and path.
        """
        self.events: list[Event] = []
        self.verdicts = verdicts or {}

    def on(self, event: Event) -> Signal:
        """Record the event and return any configured verdict.

        Args:
            event: What the runtime just did.
        """
        self.events.append(event)
        return self.verdicts.get((event.type, event.path), Signal.GO)

    def paths(self, event_type: EventType) -> list[str]:
        """Return the paths of every event of one type.

        Args:
            event_type: Type to filter on.
        """
        return [e.path for e in self.events if e.type is event_type]


def boom(_ctx):
    """Abort the run from inside a stage."""
    raise C3liRuntimeError("stage said stop")


def test_a_bare_stage_needs_no_program_wrapper():
    """`Program` is a stage, so one stage is already a program."""
    log: list[str] = []
    Runtime(Call("only", fn=lambda c: log.append("only")), hooks=[]).run()
    assert log == ["only"]


def test_run_brackets_the_program_with_run_events():
    """A trace always opens and closes."""
    hook = Record()
    Runtime(Program([Call("a", fn=lambda c: None)]), hooks=[hook]).run()
    assert hook.events[0].type is EventType.RUN_BEGAN
    assert hook.events[-1].type is EventType.RUN_ENDED
    assert hook.events[-1].payload["aborted"] is None


def test_event_paths_match_the_stage_tree():
    """Event labels, checkpoint keys and random streams share one path."""
    program = Program(
        [Call("a", fn=lambda c: None), Repeat(2, Call("b", fn=lambda c: None))]
    )
    hook = Record()
    Runtime(program, hooks=[hook]).run()
    assert hook.paths(EventType.STAGE_BEGAN) == [p for p, _ in program.walk()]


def test_a_hook_can_skip_a_stage():
    """Returning BREAK when a stage opens skips it cleanly."""
    log: list[str] = []
    program = Program(
        [
            Call("a", fn=lambda c: log.append("a")),
            Call("b", fn=lambda c: log.append("b")),
        ]
    )
    hook = Record({(EventType.STAGE_BEGAN, "program/1:b"): Signal.BREAK})
    Runtime(program, hooks=[hook]).run()
    assert log == ["a"]


def test_abort_propagates_and_still_closes_the_trace():
    """A fatal stop unwinds, but the run is still reported as ended."""
    hook = Record()
    with pytest.raises(C3liRuntimeError, match="stage said stop"):
        Runtime(Program([Call("bad", fn=boom)]), hooks=[hook]).run()
    assert hook.events[-1].type is EventType.RUN_ENDED
    assert hook.events[-1].payload["aborted"] == "stage said stop"


def test_check_rejects_an_unsatisfiable_requirement_before_any_compute():
    """A plan error must surface before the first stage runs."""
    log: list[str] = []
    program = Program(
        [
            Call("a", fn=lambda c: log.append("a")),
            Call("b", fn=lambda c: log.append("b"), requires=("missing",)),
        ]
    )
    with pytest.raises(C3liProgramError, match="requires 'missing'"):
        Runtime(program, hooks=[]).run()
    assert log == []


def test_check_accepts_a_requirement_met_by_provide():
    """Naming a shared object is how a stage asks for it."""
    program = Program(
        [Call("a", fn=lambda c: None, requires=("model",))],
        provide={"model": object()},
    )
    Runtime(program, hooks=[]).check()


def test_check_accepts_a_requirement_met_by_an_earlier_sibling():
    """What a stage publishes is available to those that follow."""
    program = Program(
        [
            Call("a", fn=lambda c: 1, provides=("score",)),
            Call("b", fn=lambda c: None, requires=("score",)),
        ]
    )
    Runtime(program, hooks=[]).check()


def test_check_rejects_a_requirement_met_only_by_a_later_sibling():
    """Order matters; a program is not a set of stages."""
    program = Program(
        [
            Call("b", fn=lambda c: None, requires=("score",)),
            Call("a", fn=lambda c: 1, provides=("score",)),
        ]
    )
    with pytest.raises(C3liProgramError, match="requires 'score'"):
        Runtime(program, hooks=[]).check()


def test_check_rejects_resume_without_a_store():
    """Both are plan errors, caught together before compute."""
    with pytest.raises(C3liProgramError, match="resume"):
        Runtime(Program([]), resume="last", hooks=[]).check()


def test_provided_modules_are_moved_to_the_device():
    """Placement is the runtime's job, so no stage calls `.to` itself."""
    model = nn.Linear(2, 2)
    seen: list[torch.device] = []
    program = Program(
        [Call("a", fn=lambda c: seen.append(next(c["model"].parameters()).device))],
        provide={"model": model},
    )
    Runtime(program, device="cpu", hooks=[]).run()
    assert seen == [torch.device("cpu")]


def test_auto_topology_is_local_without_torchrun(monkeypatch):
    """A plain `python train.py` stays single-process."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert isinstance(auto_topology(), Local)


def test_default_device_defers_to_the_torch_accelerator():
    """Covers XPU, MTIA and registered out-of-tree backends, not just CUDA."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and accelerator.is_available():
        assert default_device() == accelerator.current_accelerator()
    else:
        assert default_device() == torch.device("cpu")


def test_default_device_falls_back_without_the_accelerator_api(monkeypatch):
    """Torch below 2.6 has no `torch.accelerator`, so the probe chain runs."""
    monkeypatch.delattr(torch, "accelerator", raising=False)
    for backend in ("cuda", "xpu"):
        if getattr(torch, backend, None) is not None:
            monkeypatch.setattr(f"torch.{backend}.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert default_device() == torch.device("cpu")


def test_backends_reject_an_unknown_preset():
    """The error lists what is accepted, as elsewhere in the package."""
    with pytest.raises(ValueError, match="Unsupported backends preset"):
        apply_backends_settings("sorta")


def test_backends_default_changes_nothing():
    """`default` leaves settings alone rather than restoring torch's own."""
    was_on = torch.are_deterministic_algorithms_enabled()
    was_bench = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        apply_backends_settings("default")
        assert torch.are_deterministic_algorithms_enabled() == was_on
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = was_bench


def test_backends_fast_enables_autotuning():
    """`default` touches nothing, so autotuning needs its own preset."""
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    was_bench = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        apply_backends_settings("fast")
        assert torch.backends.cudnn.benchmark is True
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn)
        torch.backends.cudnn.benchmark = was_bench


def test_backends_strict_enables_the_torch_flags():
    """Bitwise resume on GPU needs these, so they must actually be set."""
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    was_bench = torch.backends.cudnn.benchmark
    try:
        apply_backends_settings("strict")
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn)
        torch.backends.cudnn.benchmark = was_bench


def test_trace_is_written_as_json_lines(tmp_path):
    """The trace is the oracle a resume is compared against."""
    import json

    path = tmp_path / "trace.jsonl"
    program = Program([Call("a", fn=lambda c: None)])
    Runtime(program, hooks=[Jsonl(path.name)], store=path.parent).run()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["type"] == "run.began"
    assert lines[-1]["type"] == "run.ended"
    assert [line["path"] for line in lines if line["type"] == "stage.began"] == [
        "program",
        "program/0:a",
    ]


def test_two_identical_runs_produce_identical_traces(tmp_path):
    """Nothing volatile may leak into the trace, or the oracle is useless."""

    def build():
        return Program(
            [Call("a", fn=lambda c: None), Repeat(2, Call("b", fn=lambda c: None))]
        )

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    Runtime(build(), seed=5, hooks=[Jsonl(first.name)], store=first.parent).run()
    Runtime(build(), seed=5, hooks=[Jsonl(second.name)], store=second.parent).run()
    assert first.read_text() == second.read_text()


class Breaks:
    """A hook that always raises."""

    def __init__(self, critical: bool = False):
        """Constructor.

        Args:
            critical: Whether its failure should stop the run.
        """
        self.critical = critical
        self.calls = 0

    def on(self, event: Event) -> Signal:
        """Raise on every event.

        Args:
            event: What the runtime just did.
        """
        self.calls += 1
        raise RuntimeError("hook is broken")


def test_a_failing_hook_is_dropped_not_fatal():
    """A broken console must not end a long training run."""
    broken, watcher = Breaks(), Record()
    program = Program([Call("a", fn=lambda c: None), Call("b", fn=lambda c: None)])
    with pytest.warns(UserWarning, match="dropping it"):
        Runtime(program, hooks=[broken, watcher]).run()
    assert broken.calls == 1
    assert watcher.paths(EventType.STAGE_BEGAN) == [p for p, _ in program.walk()]


def test_a_critical_hook_failing_stops_the_run():
    """A checkpoint that cannot be written must be loud, not skipped."""
    with pytest.raises(RuntimeError, match="hook is broken"):
        Runtime(
            Program([Call("a", fn=lambda c: None)]), hooks=[Breaks(critical=True)]
        ).run()


def test_a_hook_may_still_abort_the_run():
    """`C3liRuntimeError` is a RuntimeError, so isolation must not swallow it."""

    class Stopper:
        """A hook that aborts once the first stage opens."""

        def on(self, event: Event) -> Signal:
            """Abort on the first stage-began event.

            Args:
                event: What the runtime just did.
            """
            if event.type is EventType.STAGE_BEGAN:
                raise C3liRuntimeError("hook said stop")
            return Signal.GO

    with pytest.raises(C3liRuntimeError, match="hook said stop"):
        Runtime(Program([Call("a", fn=lambda c: None)]), hooks=[Stopper()]).run()


def test_muted_hooks_reset_between_runs():
    """A dropped hook gets another chance on the next run."""
    broken = Breaks()
    runtime = Runtime(Program([Call("a", fn=lambda c: None)]), hooks=[broken])
    with pytest.warns(UserWarning):
        runtime.run()
    with pytest.warns(UserWarning):
        runtime.run()
    assert broken.calls == 2


def test_store_is_resolved_to_an_absolute_path(tmp_path, monkeypatch):
    """A launcher that changes directory must not move the store."""
    monkeypatch.chdir(tmp_path)
    runtime = Runtime(Program([]), store="runs/unet", hooks=[])
    assert runtime.store.is_absolute()
    assert runtime.store == (tmp_path / "runs/unet").resolve()
    monkeypatch.chdir(tmp_path.parent)
    assert runtime.store == (tmp_path / "runs/unet").resolve()


def test_check_rejects_a_writing_hook_without_a_store():
    """Declared by attribute, so any writing hook is covered, not one by name."""

    class Writer:
        """A hook that needs somewhere to write."""

        needs_store = True

        def on(self, event: Event) -> Signal:
            """Ignore the event.

            Args:
                event: What the runtime just did.
            """
            return Signal.GO

    with pytest.raises(C3liProgramError, match="needs a store to write to"):
        Runtime(Program([]), hooks=[Writer()]).check()
    Runtime(Program([]), store="runs/x", hooks=[Writer()]).check()


def test_plan_errors_name_the_offending_value():
    """Messages show what was passed, as elsewhere in the package."""
    with pytest.raises(C3liProgramError, match=r"resume='last' needs a store"):
        Runtime(Program([]), resume="last", hooks=[]).check()


def test_auto_topology_goes_distributed_under_torchrun(monkeypatch):
    """`RANK` and `WORLD_SIZE` are how a launcher says a run is distributed."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))
    topology = auto_topology(device="cpu")
    try:
        assert isinstance(topology, Ddp)
        assert topology.is_main
        assert topology.world_size == 1
    finally:
        topology.close()


def test_device_comes_from_the_topology():
    """Only the topology knows this process's rank, so it owns the index."""
    runtime = Runtime(Program([]), hooks=[])
    assert runtime.device == runtime.topology.device


def test_an_explicit_device_overrides_the_topology():
    """A pinned device wins, index and all."""
    runtime = Runtime(
        Program([]), device="cpu", topology=Local(device="meta"), hooks=[]
    )
    assert runtime.device == torch.device("cpu")


def test_console_is_plain_for_a_non_terminal():
    """A redirected log must stay free of escape codes."""
    import io

    out = io.StringIO()
    Runtime(Program([Call("a", fn=lambda c: None)]), hooks=[Console(stream=out)]).run()
    assert "\033" not in out.getvalue()
    assert "> program/0:a" in out.getvalue()


def test_console_leaves_flags_out_of_the_numbers():
    """`bool` is an `int`, so a flag would otherwise read as a measurement."""
    import io

    out = io.StringIO()
    console = Console(stream=out)
    console.on(
        Event(
            EventType.STEP_ENDED,
            "program/0:fit",
            Progress(global_step=1),
            {"loss": 0.5, "trains": True},
        )
    )
    assert "loss=0.5" in out.getvalue()
    assert "trains" not in out.getvalue()


def test_console_colours_when_forced():
    """The override exists for terminals detection gets wrong, such as CI."""
    import io

    out = io.StringIO()
    Runtime(
        Program([Call("a", fn=lambda c: None)]),
        hooks=[Console(stream=out, color=True)],
    ).run()
    assert "\033[36m" in out.getvalue()

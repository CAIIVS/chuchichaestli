# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the benchmark declaration and its driver."""

import json
import sys

import pytest
import torch
from chuchichaestli.benchmark.benchmark import Benchmark, worker_command
from chuchichaestli.benchmark.utils import Backend, torch_input

from .benchmark_cases import BAD, GOOD, WORKER, halve, namespace, reference, square


def declare(**overrides) -> Benchmark:
    """Return a benchmark over the two test backends."""
    settings = dict(
        script="bench.py",
        backends={"good": GOOD, "bad": BAD},
        cases=lambda args: [square(8)],
        label="squares",
    )
    return Benchmark(**{**settings, **overrides})


class TestBenchmark:
    """The driver, and which mode it picks."""

    def test_it_times_reports_and_writes(self, tmp_path, capsys):
        """The plain run is a sweep, a table, and the files that were asked for."""
        path = tmp_path / "out.json"
        declare(reference=lambda case, x, args: reference(case, x)).main(namespace(json=str(path)))
        assert "8x8" in capsys.readouterr().out
        assert json.loads(path.read_text())[0]["backend"] == "good"

    def test_without_a_reference_nothing_is_checked(self, capsys):
        """A benchmark with no outside oracle still runs; it just claims less."""
        declare().main(namespace(backends=["bad"]))
        assert "MISMATCH" not in capsys.readouterr().out

    def test_it_sets_the_thread_count(self, capsys):
        """`at::parallel_for` and numpy pick different defaults, so pin it."""
        before = torch.get_num_threads()
        try:
            declare().main(namespace(threads=1))
            assert torch.get_num_threads() == 1
        finally:
            torch.set_num_threads(before)

    def test_a_gpu_run_without_a_gpu_stops(self, monkeypatch):
        """Silently falling back to the cpu would mislabel every number."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(SystemExit, match="no GPU available"):
            declare().main(namespace(device="cuda"))

    def test_the_script_defaults_to_the_one_being_run(self, monkeypatch):
        """It is re-run alongside `sys.argv[1:]`, so it has to be `sys.argv[0]`."""
        monkeypatch.setattr(sys, "argv", ["benches/dwt_impl.py", "--repeats", "3"])
        assert Benchmark(backends={}, cases=lambda args: []).script == "benches/dwt_impl.py"

    def test_an_explicit_script_still_wins(self, monkeypatch):
        """A benchmark invoked some other way says so itself."""
        monkeypatch.setattr(sys, "argv", ["pytest"])
        assert declare(script="bench.py").script == "bench.py"

    def test_a_script_that_cannot_be_re_run_is_reported(self, monkeypatch):
        """Re-running something that is not a file is a puzzling subprocess error."""
        monkeypatch.setattr(sys, "argv", ["-c"])
        with pytest.raises(SystemExit, match="cannot re-run"):
            Benchmark(backends={}, cases=lambda args: []).main(namespace(repeats=2))

    def test_a_plain_run_needs_no_script(self, capsys):
        """Only the modes that re-invoke care; the timed sweep never does."""
        declare(script="not-a-file.py").main(namespace())
        assert "8x8" in capsys.readouterr().out

    def test_it_redraws_from_saved_results_without_measuring(self, tmp_path, capsys):
        """A slow sweep is measured once; the figure can be redrawn for free."""
        saved = tmp_path / "saved.json"
        saved.write_text(json.dumps([
            {"backend": "good", "case": "8x8", "device": "cpu", "status": "ok", "median_ms": 1.0},
            {"backend": "bad", "case": "8x8", "device": "cpu", "status": "ok", "median_ms": 2.0},
        ]))
        out = tmp_path / "out.json"

        def explode(x, case):
            raise AssertionError("a redraw must not run a backend")

        args = namespace(from_json=[str(saved)], json=str(out))
        declare(backends={"good": Backend("good", explode, torch_input)}).main(args)
        assert "8x8" in capsys.readouterr().out
        assert len(json.loads(out.read_text())) == 2

    def test_a_redraw_ignores_the_device_and_threads(self, monkeypatch, tmp_path, capsys):
        """A saved result carries its own; this run measures nothing."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        saved = tmp_path / "saved.json"
        saved.write_text(json.dumps([{"backend": "good", "case": "8x8", "device": "cpu", "status": "ok", "median_ms": 1.0}]))
        before = torch.get_num_threads()
        declare().main(namespace(from_json=[str(saved)], device="cuda", threads=1))
        assert torch.get_num_threads() == before
        assert "8x8" in capsys.readouterr().out

    def test_perf_without_case_argv_says_so(self):
        """A benchmark that cannot name a case on a command line cannot count."""
        with pytest.raises(SystemExit, match="does not support --perf"):
            declare().main(namespace(perf=True))

    def test_repeats_re_runs_the_script(self, tmp_path, monkeypatch, capsys):
        """More than one repeat means fresh processes, not a longer timer."""
        script = tmp_path / "worker.py"
        script.write_text(WORKER)
        monkeypatch.setattr(sys, "argv", ["worker.py", "--scale", "1.5"])
        declare(script=str(script)).main(namespace(repeats=2))
        assert "2 runs, each in its own process" in capsys.readouterr().out

    def test_the_worker_runs_one_case_the_asked_for_number_of_times(self):
        """It is the body `perf stat` measures, so it must do nothing else."""
        calls = []
        counted = Backend("counted", lambda x, c: calls.append(1), torch_input)
        args = namespace(perf_worker=True, backends=["counted"], perf_iterations=4)
        declare(backends={"counted": counted}).main(args)
        assert len(calls) == 4

    def test_inspect_profiles_every_case_and_backend(self, capsys):
        """A profile of one backend says nothing about the one it is measured against."""
        args = namespace(profile=True, backends=["good", "bad"], profile_repeats=1, profile_rows=2)
        declare().main(args)
        out = capsys.readouterr().out
        assert "good :: 8x8" in out and "bad :: 8x8" in out

    def test_inspect_reports_a_backend_it_cannot_run(self, monkeypatch, capsys):
        """Why a backend is missing belongs in the profile output too."""
        # the reporting is what is under test, not the guard `main` runs first,
        # so the gpu is asserted present rather than left to the machine
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        cpu_only = Backend("gpuless", halve, torch_input, devices=("cpu",))
        args = namespace(profile=True, device="cuda", backends=["gpuless"])
        declare(backends={"gpuless": cpu_only}).main(args)
        assert "=== gpuless :: 8x8 === no cuda" in capsys.readouterr().out

    def test_inspect_counts_through_the_worker_command(self, tmp_path, monkeypatch, capsys):
        """`--perf` re-invokes the script per case, and reports the bandwidth."""
        import chuchichaestli.benchmark.benchmark as benchmark_module

        seen = {}

        def fake(title, command, iterations, moved_bytes=float("nan")):
            seen.update(title=title, command=command(iterations), moved=moved_bytes)

        monkeypatch.setattr(benchmark_module, "perf_case", fake)
        script = tmp_path / "worker.py"
        script.write_text(WORKER)
        args = namespace(perf=True, perf_iterations=50)
        declare(
            script=str(script),
            case_argv=lambda case: ["--side", "8"],
            moved_bytes=lambda case: 1024.0,
        ).main(args)
        assert seen["title"] == "good :: 8x8"
        command = seen["command"]
        assert "--side" in command
        assert command[command.index("--perf-iterations") + 1] == "50"
        assert seen["moved"] == 1024.0


class TestWorkerCommand:
    """The command line one case is handed to a fresh process on."""

    def test_it_carries_the_shared_arguments_and_the_case(self):
        """The worker has to reconstruct the very case that was asked for."""
        args = namespace(threads=None)
        command = worker_command("bench.py", GOOD.name, args, ["--side", "8"])(300)
        assert command[1:] == [
            "bench.py", "--perf-worker", "--backends", "good", "--device", "cpu",
            "--dtype", "float32", "--side", "8", "--perf-iterations", "300",
        ]

    def test_threads_are_passed_on_only_when_pinned(self):
        """An unset thread count must stay unset, not become the torch default."""
        assert "--threads" not in worker_command("b.py", GOOD.name, namespace(threads=None), [])(1)
        assert "--threads" in worker_command("b.py", GOOD.name, namespace(threads=1), [])(1)

    def test_the_iteration_count_is_what_varies(self):
        """The idle run is the same command with nothing to do."""
        command = worker_command("b.py", GOOD.name, namespace(threads=None), [])
        assert command(0)[-1] == "0" and command(200)[-1] == "200"

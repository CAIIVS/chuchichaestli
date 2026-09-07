# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the benchmark profiling utilities."""

import subprocess
import sys

import torch
from chuchichaestli.benchmark.profile import EVENTS, perf_case, perf_counters, profile_case


# `perf stat -x ,`: value,unit,event,enabled ns,percent running,metric,metric unit
PERF_OUTPUT = """1234567890,,cycles:u,1500000000,100.00,,
2000000000,,instructions:u,1500000000,100.00,,
50000000,,cache-references:u,1500000000,100.00,,
1000000,,cache-misses:u,1500000000,100.00,,
"""

IDLE_OUTPUT = """234567890,,cycles:u,500000000,100.00,,
1000000000,,instructions:u,500000000,100.00,,
10000000,,cache-references:u,500000000,100.00,,
100000,,cache-misses:u,500000000,100.00,,
"""


def fake_run(stderr: str, returncode: int = 0):
    """Return a `subprocess.run` that answers with this `perf stat` output."""

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    return run


class TestProfileCase:
    """Attributing the time to operators."""

    def test_it_prints_a_table_naming_the_case(self, capsys):
        """The header is what tells two profiles of one run apart."""
        x = torch.randn(64, 64)
        profile_case(lambda: x @ x, "matmul :: 64x64", "cpu", repeats=2, rows=3, warmup=1)
        out = capsys.readouterr().out
        assert "=== matmul :: 64x64 ===" in out
        assert "Self CPU" in out

    def test_it_calls_the_work_warmup_plus_repeats_times(self, capsys):
        """Lazy set-up is warmed up first so it is not attributed to the work."""
        calls = []
        profile_case(lambda: calls.append(1), "counting", "cpu", repeats=3, warmup=2)
        assert len(calls) == 5


    def test_it_writes_a_trace_when_asked(self, tmp_path, capsys):
        """A timeline needs the trace file; the table alone cannot show overlap."""
        import json

        x = torch.randn(32, 32)
        profile_case(lambda: x @ x, "matmul :: 32x32", "cpu", repeats=2, rows=3, trace=str(tmp_path))
        written = list(tmp_path.glob("*.json"))
        assert len(written) == 1
        assert written[0].name == "matmul-32x32.json"
        events = json.loads(written[0].read_text())
        assert events["traceEvents"] if isinstance(events, dict) else events

    def test_it_writes_nothing_without_a_trace_directory(self, tmp_path, capsys):
        """Profiling should not litter the tree by default."""
        x = torch.randn(32, 32)
        profile_case(lambda: x @ x, "matmul :: 32x32", "cpu", repeats=2, rows=3)
        assert list(tmp_path.iterdir()) == []

    def test_each_case_gets_its_own_file(self, tmp_path, capsys):
        """A sweep profiles several cases; one must not overwrite the next."""
        x = torch.randn(32, 32)
        for title in ("a :: 1d haar", "b :: 2d db4"):
            profile_case(lambda: x @ x, title, "cpu", repeats=1, rows=3, trace=str(tmp_path))
        assert sorted(p.name for p in tmp_path.glob("*.json")) == ["a-1d-haar.json", "b-2d-db4.json"]


class TestPerfCounters:
    """Reading what `perf stat` wrote to stderr."""

    def test_it_parses_counters_and_the_wall_time(self, monkeypatch):
        """The counters are the numbers; the wall time is a separate line."""
        monkeypatch.setattr(subprocess, "run", fake_run(PERF_OUTPUT))
        counters, seconds = perf_counters(["python", "worker.py"])
        assert counters["cycles:u"] == 1234567890
        assert counters["cache-misses:u"] == 1000000
        assert seconds == 1.5

    def test_the_time_is_the_counters_enabled_time(self, monkeypatch):
        """The rates are derived over it, so it has to be what perf measured."""
        monkeypatch.setattr(subprocess, "run", fake_run(IDLE_OUTPUT))
        _, seconds = perf_counters(["python", "worker.py"])
        assert seconds == 0.5

    def test_a_counter_the_machine_cannot_provide_is_left_out(self, monkeypatch):
        """A missing counter is unknown, which is not the same as zero."""
        stderr = PERF_OUTPUT + "<not supported>,,cycles:k,0,100.00,,\n"
        monkeypatch.setattr(subprocess, "run", fake_run(stderr))
        counters, _ = perf_counters(["python", "worker.py"])
        assert "cycles:k" not in counters and counters["cycles:u"] == 1234567890

    def test_it_does_not_depend_on_the_locale(self, monkeypatch):
        """The human-readable output groups digits; `-x` is why we do not read it."""
        human = "     1'234'567'890      cycles:u\n       1.500000000 seconds time elapsed\n"
        monkeypatch.setattr(subprocess, "run", fake_run(human))
        assert perf_counters(["python", "worker.py"]) == ({}, 0.0)

    def test_it_survives_output_with_no_counters(self, monkeypatch):
        """A locked-down machine reports nothing rather than crashing."""
        monkeypatch.setattr(subprocess, "run", fake_run("no permission\n"))
        assert perf_counters(["true"]) == ({}, 0.0)

    def test_it_runs_the_command_under_perf_with_the_events(self, monkeypatch):
        """The events asked for are the events measured."""
        seen = {}

        def run(command, **kwargs):
            seen["command"] = command
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(subprocess, "run", run)
        perf_counters(["python", "worker.py"], events="cycles:u")
        assert seen["command"] == [
            "perf", "stat", "-x", ",", "-e", "cycles:u", "python", "worker.py"
        ]


class TestPerfCase:
    """The counters of the work, with the interpreter's start-up taken off."""

    def test_it_subtracts_the_idle_run(self, monkeypatch, capsys):
        """Importing torch costs more than the loop, so it must not be counted."""
        outputs = iter([PERF_OUTPUT, IDLE_OUTPUT])

        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "", next(outputs))

        monkeypatch.setattr(subprocess, "run", run)
        perf_case("case", lambda n: [sys.executable, "-c", "pass", str(n)], 100)
        out = capsys.readouterr().out
        assert "1,000,000,000" in out  # 1,234,567,890 - 234,567,890

    def test_it_reports_the_derived_rates(self, monkeypatch, capsys):
        """A raw counter says little; the ratios are what a run is read by."""
        outputs = iter([PERF_OUTPUT, IDLE_OUTPUT])
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", next(outputs))
        )
        perf_case("case", lambda n: ["true", str(n)], 100)
        out = capsys.readouterr().out
        assert "instructions per cycle" in out
        assert "cache miss rate" in out
        assert "per call" in out

    def test_bandwidth_is_reported_only_when_the_traffic_is_known(self, monkeypatch, capsys):
        """A benchmark that cannot size its own traffic still gets its counters."""
        outputs = iter([PERF_OUTPUT, IDLE_OUTPUT])
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", next(outputs))
        )
        perf_case("case", lambda n: ["true"], 100)
        out = capsys.readouterr().out
        assert "per call" in out and "achieved bandwidth" not in out

        outputs = iter([PERF_OUTPUT, IDLE_OUTPUT])
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", next(outputs))
        )
        perf_case("case", lambda n: ["true"], 100, moved_bytes=1e6)
        assert "achieved bandwidth" in capsys.readouterr().out

    def test_a_counter_under_the_idle_run_is_not_reported_as_a_number(self, monkeypatch, capsys):
        """Cache misses at a low iteration count are start-up noise, not work."""
        noisy = IDLE_OUTPUT.replace("100000,,cache-misses:u", "9000000,,cache-misses:u")
        outputs = iter([PERF_OUTPUT, noisy])
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", next(outputs))
        )
        perf_case("case", lambda n: ["true"], 100)
        out = capsys.readouterr().out
        assert "under the idle run" in out
        assert "need more than 100 iterations" in out
        # the rate over a counter that never rose above the noise would be negative
        assert "cache miss rate" not in out
        assert "-" not in out.split("cache-misses:u")[1].splitlines()[0]

    def test_the_rates_still_appear_for_the_counters_that_did_rise(self, monkeypatch, capsys):
        """One noisy counter must not take the rest of the report down with it."""
        noisy = IDLE_OUTPUT.replace("100000,,cache-misses:u", "9000000,,cache-misses:u")
        outputs = iter([PERF_OUTPUT, noisy])
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", next(outputs))
        )
        perf_case("case", lambda n: ["true"], 100)
        out = capsys.readouterr().out
        assert "instructions per cycle" in out and "per call" in out

    def test_zero_iterations_does_not_divide_by_them(self, monkeypatch, capsys):
        """The idle run passes 0 to the worker, so the guard cannot be in the flag."""
        outputs = iter([PERF_OUTPUT, IDLE_OUTPUT])
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", next(outputs))
        )
        perf_case("case", lambda n: ["true"], 0, moved_bytes=1024.0)
        out = capsys.readouterr().out
        assert "per call" not in out and "achieved bandwidth" not in out

    def test_a_missing_perf_is_reported_not_raised(self, monkeypatch, capsys):
        """`perf` is not installed everywhere; that is not a reason to stop."""

        def run(command, **kwargs):
            raise FileNotFoundError("no perf")

        monkeypatch.setattr(subprocess, "run", run)
        perf_case("case", lambda n: ["true"], 10)
        assert "perf could not run" in capsys.readouterr().out

    def test_no_counters_hints_at_the_usual_cause(self, monkeypatch, capsys):
        """The answer is almost always `perf_event_paranoid`, so say so."""
        monkeypatch.setattr(
            subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", "")
        )
        perf_case("case", lambda n: ["true"], 10)
        assert "perf_event_paranoid" in capsys.readouterr().out

    def test_the_default_events_are_the_four_that_are_read_back(self, monkeypatch):
        """Every counter the report derives a rate from has to be collected."""
        assert set(EVENTS.split(",")) == {
            "cycles:u", "instructions:u", "cache-references:u", "cache-misses:u"
        }

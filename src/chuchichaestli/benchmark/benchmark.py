# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The benchmark driver that runs timings, performance measurements, etc.

A benchmark names its backends, its cases and how it is checked, builds a
`Benchmark` out of them, and its `main` is then two lines:
```python
    BENCHMARK = Benchmark(script=__file__, backends=BACKENDS, cases=build_cases)

    def main(argv=None):
        BENCHMARK.main(parse(argv))
```
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from chuchichaestli.benchmark.profile import perf_case, profile_case
from chuchichaestli.benchmark.utils import (
    Backend,
    Case,
    load_rows,
    report,
    report_rows,
    repeated,
    sweep,
    write,
    write_rows,
)


def worker_command(
    script: str, backend: str, args, case_argv: Sequence[str]
) -> Callable[[int], list[str]]:
    """Return a command running one case a given number of times, and nothing else.

    `perf stat` measures a process, so `--perf-worker` runs just this loop in a
    fresh one; the shared arguments are filled in here.

    Args:
        script: Benchmark to run, normally the caller's `__file__`.
        backend: Name of the backend to run.
        args: Parsed command line arguments.
        case_argv: Flags selecting the one case to run.
    """

    def command(iterations: int) -> list[str]:
        line = [
            sys.executable, script, "--perf-worker",
            "--backends", backend,
            "--device", args.device,
            "--dtype", args.dtype,
            *case_argv,
            "--perf-iterations", str(iterations),
        ]
        if args.threads is not None:
            line += ["--threads", str(args.threads)]
        return line

    return command


@dataclass
class Benchmark:
    """The driver that runs a benchmark for backends.

    Provides a `main` function for parsed command-line arguments.
    These arguments select the mode path:
        - timed sweep (default)
        - profiling (`args.profile`)
        - hardware counters (`args.perf`)

    Args:
        backends: Every backend that exists, by name.
        cases: Expands the parsed command line into the cases of the sweep.
        script: The benchmark to re-invoke for the repeated sweep and for the
            `perf stat` worker; default is the script this process is running.
        label: Names the whole comparison in the printed table.
        reference: Returns the arrays every backend is checked against, given a
            case, its input and the arguments; nothing is checked without one.
        group: Bars sharing a value of this are drawn side by side.
        case_argv: Flags naming one case, for the `perf stat` worker; a
            benchmark without one cannot run `--perf`.
        moved_bytes: Bytes one run of a case has to move, for the achieved
            bandwidth; omitted from the counter report without one.
    """

    backends: Mapping[str, Backend]
    cases: Callable[[Any], Sequence[Case]]
    script: str = field(default_factory=lambda: sys.argv[0])
    label: str = "benchmark"
    reference: Callable[[Case, torch.Tensor, Any], list | None] | None = None
    group: Callable[[dict], str] | None = None
    case_argv: Callable[[Case], list[str]] | None = None
    moved_bytes: Callable[[Case], float] | None = None

    def worker(self, args) -> None:
        """Run one case and nothing else, which is what `perf stat` measures.

        Args:
            args: Parsed command line arguments.
        """
        case = self.cases(args)[0]
        backend = self.backends[args.backends[0]]
        payload = backend.prepare(case.sample().to(args.device), case)
        for _ in range(args.perf_iterations):
            backend.apply(payload, case)

    def inspect(self, args) -> None:
        """Profile or count, rather than time, every case of the sweep.

        Args:
            args: Parsed command line arguments.
        """
        for case in self.cases(args):
            for name in args.backends:
                backend = self.backends[name]
                status = backend.probe(args.device, case)
                title = f"{name} :: {case.label()}"
                if status != "ok":
                    print(f"\n=== {title} === {status}")
                    continue
                if args.profile:
                    payload = backend.prepare(case.sample().to(args.device), case)
                    profile_case(
                        functools.partial(backend.apply, payload, case),
                        title,
                        args.device,
                        args.profile_repeats,
                        args.profile_rows,
                    )
                if args.perf:
                    perf_case(
                        title,
                        worker_command(self.script, name, args, self.case_argv(case)),
                        args.perf_iterations,
                        self.moved_bytes(case) if self.moved_bytes else float("nan"),
                    )

    def measure(self, args) -> None:
        """Time every backend over every case, then report and write.

        Args:
            args: Parsed command line arguments.
        """
        cases = self.cases(args)
        if self.reference is None:
            results = sweep(cases, self.backends, args, label=self.label)
        else:
            reference = functools.partial(self.reference, args=args)
            results = sweep(cases, self.backends, args, reference, self.label)
        report(results, args)
        write(results, args, self.group)

    def redraw(self, args) -> None:
        """Report and plot saved results, measuring nothing.

        Args:
            args: Parsed command line arguments.
        """
        rows = load_rows(args.from_json)
        report_rows(rows)
        write_rows(rows, args, self.group)

    def main(self, args) -> None:
        """Run whichever of the modes the command line asked for.

        Args:
            args: Parsed command line arguments.

        Raises:
            SystemExit: If a GPU run is asked for on a machine without one, or
                if `--perf` is asked of a benchmark that cannot name its cases
                on a command line.
        """
        # nothing is measured, so neither the device nor the thread count is
        # this run's to have; a saved result carries its own
        if args.from_json:
            self.redraw(args)
            return

        if args.threads is not None:
            torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("no GPU available")
        if args.perf and self.case_argv is None:
            raise SystemExit(f"{self.label} does not support --perf: it supplies no case_argv")
        # both of these re-run the benchmark (using `script`)
        if (args.repeats > 1 or args.perf) and not Path(self.script).is_file():
            raise SystemExit(f"cannot re-run {self.label} as {self.script!r}; pass script= explicitly")
        if args.perf_worker:
            self.worker(args)
        elif args.profile or args.perf:
            self.inspect(args)
        elif args.repeats > 1:
            repeated(self.script, args)
        else:
            self.measure(args)

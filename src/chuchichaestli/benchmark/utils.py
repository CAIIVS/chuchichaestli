# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The measuring machinery the benchmarks share.

This module provides utilities that inform a benchmark about tests (`Case`),
execution (`Backends`), and results (`Result`). Moreover, utility functions
that produce, consume, or report these components:
```python
    cases = [MyCase(...)]
    results = sweep(cases, BACKENDS, args, sample)
    report(results, args)
    write(results, args)
```
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
import torch.utils.benchmark as benchmark

from chuchichaestli.benchmark.args import without_options


@runtime_checkable
class Case(Protocol):
    """One point of a sweep, whatever the benchmark varies over."""

    def label(self) -> str:
        """Short description, used as the row name in every report."""
        ...

    def fields(self) -> dict[str, Any]:
        """The case as scalars, for the json and csv output."""
        ...

    def sample(self) -> torch.Tensor:
        """Draw the input, identically for every backend."""
        ...


@dataclass(frozen=True, kw_only=True)
class TensorCase:
    """A sweep point whose input is one drawn tensor.

    A benchmark subclasses this with whatever else its sweep varies, and gets
    the input, the output columns and the sizes automatically.

    Args:
        shape: Extent of the axes the sweep varies, without the leading axes.
        dtype: Working dtype, pinned for the whole run.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    lead: ClassVar[tuple[int, ...]] = (2, 1)
    seed: ClassVar[int] = 0

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Shape of the drawn input, leading axes included."""
        return (*self.lead, *self.shape)

    @property
    def elements(self) -> int:
        """Number of elements one input holds."""
        return math.prod(self.tensor_shape)

    @property
    def nbytes(self) -> int:
        """Bytes one input occupies."""
        return self.elements * torch.empty((), dtype=self.dtype).element_size()

    def sample(self) -> torch.Tensor:
        """Draw the input, identically for every backend and every run."""
        generator = torch.Generator().manual_seed(self.seed)
        return torch.randn(self.tensor_shape, dtype=self.dtype, generator=generator)

    def label(self) -> str:
        """Short description, used as the row name in every report."""
        return "x".join(str(n) for n in self.shape)

    @staticmethod
    def render(value: Any) -> Any:
        """Return one field of a case as something json and csv can hold.

        Override to teach a benchmark's own field types how to be a column.

        Args:
            value: Field to render.
        """
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, (tuple, list)):
            return "x".join(str(n) for n in value)
        return value

    def fields(self) -> dict[str, Any]:
        """Every field of the case, rendered for the json and csv output."""
        return {f.name: self.render(getattr(self, f.name)) for f in dataclasses.fields(self)}


@dataclass
class Backend:
    """One implementation to be tested.

    Args:
        name: Name the results are reported under.
        apply: Does the work being measured, given a prepared input and a case.
        prepare: Puts a sample tensor into the form `apply` takes.
        to_arrays: Puts a result into the canonical order, as numpy arrays; the
            default assumes `apply` already returns a flat sequence of tensors.
        devices: Devices the backend can run on.
        supports: Returns why this backend cannot run a case, or `''` if it can.
        note: Caveat printed with the results.
        differentiable: Whether a backward pass can be timed at all.
    """

    name: str
    apply: Callable[[Any, Case], Any]
    prepare: Callable[[torch.Tensor, Case], Any]
    to_arrays: Callable[[Any], list] | None = None
    devices: tuple[str, ...] = ("cpu", "cuda")
    supports: Callable[[Case], str] = lambda case: ""  # assume always supported
    note: str = ""
    differentiable: bool = True

    def arrays(self, result: Any) -> list:
        """Return a result in the canonical order, as numpy arrays.

        Args:
            result: Result in whatever form the backend returned.
        """
        if self.to_arrays is not None:
            return self.to_arrays(result)
        return [band.detach().cpu().numpy() for band in result]

    def probe(self, device: str, case: Case) -> str:
        """Return `'ok'`, or why this backend cannot run a case.

        Args:
            device: Device to run on.
            case: Case to run.
        """
        if device not in self.devices:
            return f"no {device}"
        if reason := self.supports(case):
            return reason
        try:
            x = torch.zeros_like(case.sample())
            self.apply(self.prepare(x, case), case)
        except ImportError:
            return "not installed"
        except Exception as exc:  # noqa: BLE001 - any failure disqualifies the backend
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return reason[:44]
        return "ok"

    def check(
        self, device: str, case: Case, x: torch.Tensor, reference: list | None, dtype: torch.dtype
    ) -> bool:
        """Return whether this backend agrees with the reference on a case.

        Args:
            device: Device to run on.
            case: Case to run.
            x: The very input the reference was computed from.
            reference: Reference arrays, or `None` to skip the check.
            dtype: Working dtype, which sets how tight the gate can be.
        """
        if reference is None:
            return True
        import numpy as np

        ours = self.arrays(self.apply(self.prepare(x.to(device), case), case))
        if len(ours) != len(reference):
            return False
        # values near zero, where a relative tolerance buys nothing, so the gate
        # has to be loose enough for the rounding of the working dtype
        atol = 1e-10 if dtype is torch.float64 else 1e-4
        return all(
            a.shape == b.shape and np.allclose(a, b, rtol=1e-4, atol=atol)
            for a, b in zip(ours, reference, strict=True)
        )

    def timed_call(self, payload: Any, case: Case, direction: str) -> Callable[[], Any]:
        """Return the callable the timer runs.

        Args:
            payload: Input in whatever form this backend takes.
            case: Case to run.
            direction: `'forward'` or `'backward'`.
        """
        if direction == "forward":
            return lambda: self.apply(payload, case)

        def backward() -> None:
            total_energy(self.apply(payload, case)).backward()

        return backward


@dataclass
class Result:
    """One measurement, or the reason there is none."""

    backend: str
    device: str
    case: Case
    status: str = "ok"
    measurement: benchmark.Measurement | None = None
    peak_mb: float = float("nan")

    @property
    def median_ms(self) -> float:
        """Median run time in milliseconds, or `nan` if there is no measurement."""
        return float("nan") if self.measurement is None else self.measurement.median * 1e3

    @property
    def iqr_ms(self) -> float:
        """Interquartile range in milliseconds, or `nan` if there is no measurement."""
        return float("nan") if self.measurement is None else self.measurement.iqr * 1e3

    def row(self) -> dict[str, Any]:
        """The result as flat scalars, for the json and csv output.

        `label` and `threads` come along because they are what `Compare` groups
        its tables by, so a saved run can be reported the way it was measured.
        """
        spec = self.measurement.task_spec if self.measurement is not None else None
        return {
            "backend": self.backend,
            "device": self.device,
            "case": self.case.label(),
            **self.case.fields(),
            "status": self.status,
            "median_ms": self.median_ms,
            "iqr_ms": self.iqr_ms,
            "peak_mb": self.peak_mb,
            "label": spec.label if spec else None,
            "threads": spec.num_threads if spec else None,
        }


def torch_input(x: torch.Tensor, case: Case) -> torch.Tensor:
    """Return the input a torch backend takes.

    Args:
        x: Sample drawn for the case.
        case: Case being run.
    """
    return x


def numpy_input(x: torch.Tensor, case: Case):
    """Return the input a numpy backend takes.

    Args:
        x: Sample drawn for the case.
        case: Case being run.
    """
    return x.detach().cpu().numpy()


def peak_memory(call: Callable[[], Any], device: str) -> float:
    """Return the peak allocation of one call, in mebibytes.

    Reported on the GPU only: on the CPU the caching allocator hides torch
    tensors from `tracemalloc`, which would flatter torch and mislead for numpy.

    Args:
        call: Callable to measure.
        device: Device the call runs on.
    """
    if device != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    call()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def total_energy(result: Any) -> torch.Tensor:
    """Sum every tensor of a result, whatever shape the structure has.

    Walks the structure the backend returned rather than the canonical arrays,
    which are detached and so carry no graph to run a backward pass through.

    Args:
        result: Result to sum.
    """
    if torch.is_tensor(result):
        return result.sum()
    values = result.values() if isinstance(result, Mapping) else result
    return sum(total_energy(value) for value in values)


def sweep(
    cases: Sequence[Case],
    backends: Mapping[str, Backend],
    args,
    reference: Callable[[Case, torch.Tensor], list | None] = lambda case, x: None,
    label: str = "benchmark",
) -> list[Result]:
    """Run every backend over every case and return one result each.

    A backend is probed, then checked against the reference, and only then
    timed: a fast wrong answer is not a result.

    Args:
        cases: Cases of the sweep.
        backends: Every backend that exists, by name.
        args: Parsed command line arguments.
        reference: Returns the arrays every backend is checked against, or
            `None` for a case that has no reference.
        label: Names the whole comparison in the printed table.
    """
    direction = getattr(args, "direction", "forward")
    dtype = getattr(torch, args.dtype)
    results: list[Result] = []

    for case in cases:
        x = case.sample()
        expected = reference(case, x)

        for name in args.backends:
            backend = backends[name]
            status = backend.probe(args.device, case)
            if status != "ok":
                results.append(Result(name, args.device, case, status))
                continue
            if not backend.check(args.device, case, x, expected, dtype):
                results.append(Result(name, args.device, case, "MISMATCH"))
                continue
            if direction == "backward" and not backend.differentiable:
                results.append(Result(name, args.device, case, "n/a"))
                continue

            payload = backend.prepare(x.to(args.device), case)
            if direction == "backward":
                payload = payload.requires_grad_(True)
            call = backend.timed_call(payload, case, direction)

            peak = peak_memory(call, args.device)
            timer = benchmark.Timer(
                stmt="call()",
                globals={"call": call},
                num_threads=args.threads or torch.get_num_threads(),
                label=f"{label} ({direction}, {args.device})",
                sub_label=case.label(),
                description=name,
            )
            results.append(
                Result(
                    name,
                    args.device,
                    case,
                    "ok",
                    timer.blocked_autorange(min_run_time=args.min_run_time),
                    peak,
                )
            )
    return results


def load_rows(paths: Sequence[str]) -> list[dict]:
    """Read result rows from saved json files, merged into one list.

    Rows are the same measurement only when they agree on the device and on
    everything `Compare` lays a table out by; where they collide the later wins.

    Args:
        paths: Files to read, in increasing order of precedence.

    Raises:
        SystemExit: If a file is missing or is not the json a run writes.
    """
    merged: dict[tuple, dict] = {}
    superseded = 0
    for path in paths:
        try:
            with open(path) as fh:
                rows = json.load(fh)
        except FileNotFoundError:
            raise SystemExit(f"no such result file: {path}") from None
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path} is not valid json: {exc}") from None
        if not isinstance(rows, list):
            raise SystemExit(f"{path} does not hold a list of result rows")
        for row in rows:
            if not isinstance(row, dict) or not {"backend", "case", "status"} <= row.keys():
                raise SystemExit(f"{path} holds something other than result rows")
            # the device is what separates rows written before `label` and
            # `threads` were recorded, where both fall back to None
            key = (
                row["backend"], row["case"], row.get("device"),
                row.get("label"), row.get("threads"),
            )
            superseded += key in merged
            merged[key] = row
    if superseded:
        print(f"{superseded} row(s) measured more than once; kept the later file")
    return list(merged.values())


def as_measurements(rows: Sequence[dict]) -> list[benchmark.Measurement]:
    """Rebuild timer measurements from saved rows, so `Compare` can lay them out.

    Only the median survives json, which is what `Compare` reports anyway; a row
    written before `label` and `threads` falls back to the device at one thread.

    Args:
        rows: Rows to rebuild from.
    """
    rebuilt = []
    for row in rows:
        if row["status"] != "ok" or row.get("median_ms") is None:
            continue
        spec = benchmark.TaskSpec(
            stmt="",
            setup="",
            label=row.get("label") or f"saved results ({row.get('device', 'cpu')})",
            sub_label=row["case"],
            description=row["backend"],
            num_threads=row.get("threads") or 1,
        )
        rebuilt.append(benchmark.Measurement(1, [row["median_ms"] / 1e3], spec))
    return rebuilt


def report_rows(rows: Sequence[dict]) -> None:
    """Print already flattened result rows the way a live run prints them.

    Args:
        rows: Rows to print.
    """
    print()
    measured = as_measurements(rows)
    if measured:
        comparison = benchmark.Compare(measured)
        comparison.colorize(rowwise=True)
        comparison.print()

    skipped = [row for row in rows if row["status"] != "ok"]
    if skipped:
        print("\nnot measured")
        for row in sorted(skipped, key=lambda r: (r.get("device", ""), r["case"], r["backend"])):
            device = row.get("device", "")
            print(f"  {device:5s} {row['case']:34s} {row['backend']:12s} {row['status']}")


def report(results: Sequence[Result], args) -> None:
    """Print the measurements, and why any backend was left out.

    Args:
        results: Results to print.
        args: Parsed command line arguments.
    """
    print()

    measured = [r.measurement for r in results if r.measurement is not None]
    if measured:
        comparison = benchmark.Compare(measured)
        comparison.colorize(rowwise=True)
        comparison.print()

    if args.device == "cuda" and measured:
        print("\npeak memory / MiB")
        for r in results:
            if r.measurement is not None:
                print(f"  {r.case.label():34s} {r.backend:9s} {r.peak_mb:8.2f}")

    skipped = [r for r in results if r.measurement is None]
    if skipped:
        print("\nnot measured")
        for r in skipped:
            print(f"  {r.case.label():34s} {r.backend:9s} {r.status}")


def write(results: Sequence[Result], args, group: Callable[[dict], str] | None = None) -> None:
    """Write the results to the requested files.

    Args:
        results: Results to write.
        args: Parsed command line arguments.
        group: Bars sharing a value of this are drawn side by side; the whole
            case label if omitted.
    """
    write_rows([r.row() for r in results], args, group)


def write_rows(rows: Sequence[dict], args, group: Callable[[dict], str] | None = None) -> None:
    """Write already flattened result rows to the requested files.

    Args:
        rows: Rows to write.
        args: Parsed command line arguments.
        group: Bars sharing a value of this are drawn side by side; the whole
            case label if omitted.
    """
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    if args.plot:
        plot(rows, args.plot, group)


def plot(
    rows: Sequence[dict],
    path: str,
    group: Callable[[dict], str] | None = None,
    palette: Sequence[str] = (
        "#60293F", "#964063", "#EA9739", "#9BF1F7", "#222B4F"
    ),
) -> None:
    """Draw the timings as a grouped bar chart.

    Args:
        rows: Result rows to draw.
        path: File to write the figure to.
        group: Bars sharing a value of this are drawn side by side; the whole
            case label if omitted.
        palette: Bar colours, cycled when there are more backends than colours;
            the stops of the header gradient in `docs/stylesheets/extra.css` by
            default, so a figure dropped into the docs belongs there.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = group or (lambda row: row["case"])
    timed = [row for row in rows if row["status"] == "ok"]
    if not timed:
        print("nothing to plot")
        return
    groups = sorted({group(row) for row in timed})
    backends = sorted({row["backend"] for row in timed})
    width = 0.8 / len(backends)
    figure, axes = plt.subplots(figsize=(2 + 1.4 * len(groups), 4))
    for i, backend in enumerate(backends):
        heights = [
            next(
                (row["median_ms"] for row in timed if row["backend"] == backend and group(row) == name),
                0.0,
            )
            for name in groups
        ]
        axes.bar(
            [x + i * width for x in range(len(groups))],
            heights,
            width,
            label=backend,
            color=palette[i % len(palette)],
        )
    axes.set_xticks([x + 0.4 - width / 2 for x in range(len(groups))])
    axes.set_xticklabels(groups, rotation=30, ha="right")
    axes.set_ylabel("median time / ms")
    axes.set_yscale("log")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    print(f"wrote {path}")


# Increase `glibc`s mmap and heap trim threshold to 128 MiB
ALLOCATOR_ENV = {
    "MALLOC_MMAP_THRESHOLD_": "134217728",
    "MALLOC_TRIM_THRESHOLD_": "134217728",
}


def pin_allocator() -> None:
    """Re-exec once with the allocator thresholds pinned.

    glibc reads them when it initialises, so a process cannot set them for
    itself; the exec is what makes them take effect.
    """
    if all(os.environ.get(key) == value for key, value in ALLOCATOR_ENV.items()):
        return
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        {**os.environ, **ALLOCATOR_ENV},
    )


def repeated(
    script: str, args, group: Callable[[dict], str] | None = None
) -> None:
    """Measure the sweep several times over, each in its own process.

    A timer only sees its own process, so fresh ones expose a case whose speed
    was settled at startup; the spread is whatever `pin_allocator` leaves.

    Args:
        script: Benchmark to run, normally the caller's `__file__`.
        args: Parsed command line arguments.
        group: Bars sharing a value of this are drawn side by side; the whole
            case label if omitted.

    Raises:
        SystemExit: If any of the runs fails, since a missing run would
            silently narrow the spread the others report.
    """
    # a child measures once and writes json; the flags that would make it
    # repeat, or write the caller's files, stay behind
    trimmed = without_options(sys.argv[1:], {"--repeats", "--json", "--csv", "--plot"})

    runs: list[list[dict]] = []
    for attempt in range(args.repeats):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            path = handle.name
        command = [sys.executable, script, *trimmed, "--json", path]
        finished = subprocess.run(command, capture_output=True, text=True)
        if finished.returncode != 0:
            print(finished.stdout[-2000:])
            raise SystemExit(f"run {attempt + 1} failed:\n{finished.stderr[-2000:]}")
        with open(path) as fh:
            runs.append(json.load(fh))
        os.unlink(path)

    merged: dict[tuple[str, str], list[float]] = {}
    labels: dict[tuple[str, str], dict] = {}
    skipped: dict[tuple[str, str], dict] = {}
    for rows in runs:
        for row in rows:
            key = (row["backend"], row["case"])
            if row["status"] != "ok" or row["median_ms"] is None:
                skipped.setdefault(key, row)
                continue
            merged.setdefault(key, []).append(row["median_ms"])
            labels[key] = row

    rows_out = []
    for backend, case in sorted(merged):
        times = sorted(merged[backend, case])
        row = dict(labels[backend, case])
        row["median_ms"] = times[len(times) // 2]
        row["spread"] = max(times) / min(times) if min(times) else float("nan")
        row["runs"] = times
        rows_out.append(row)
    # a backend that could not run is worth the same line here as anywhere else
    rows_out += [skipped[key] for key in sorted(skipped) if key not in merged]

    print(f"\n{args.repeats} runs, each in its own process")
    report_rows(rows_out)

    spreads = sorted(row["spread"] for row in rows_out if "spread" in row)
    if spreads:
        middle = spreads[len(spreads) // 2]
        unstable = sorted(
            (row for row in rows_out if row.get("spread", 0) > 1.15),
            key=lambda row: -row["spread"],
        )
        print(f"\nspread across runs: median {middle:.2f}x, {len(unstable)} of {len(spreads)} above 1.15x")
        for row in unstable:
            print(f"  {row['case']:34s} {row['backend']:12s} {row['spread']:5.2f}x")

    write_rows(rows_out, args, group)

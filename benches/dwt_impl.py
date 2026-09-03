# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark the wavelet transform backends against each other and external libraries.

Every backend is checked against PyWavelets before it is timed, so a fast wrong
answer cannot win. The comparison libraries are not dependencies; pull them in
for the run, and use one thread to be fair to PyWavelets, which is
single-threaded C on numpy:

    uv run --with pywavelets --with ptwt --with pytorch_wavelets python benches/dwt_impl.py --threads 1

Drop `--threads 1` for what the torch backends actually get. Further examples:

    # the torch libraries on the GPU, where PyWavelets cannot follow
    python benches/dwt_impl.py --device cuda --backends torch ptwt ptwavelets

    # one case, backward, in double precision
    python benches/dwt_impl.py --dims 2 --sizes 512x512 --levels 3 --dtype float64 --direction backward

    # keep the numbers
    python benches/dwt_impl.py --json out.json --csv out.csv --plot out.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.utils.benchmark as benchmark

from chuchichaestli.dwt.functional import wavedecn


@dataclass(frozen=True)
class Case:
    """One point of the sweep."""

    dimensions: int
    wavelet: str
    mode: str
    levels: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def axes(self) -> tuple[int, ...]:
        """Axes the transform runs over."""
        return tuple(range(-self.dimensions, 0))

    def label(self) -> str:
        """Short description for the result table."""
        return (
            f"{self.dimensions}d {self.wavelet} {self.mode} L{self.levels} "
            f"{'x'.join(str(n) for n in self.shape)}"
        )


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


def flatten(coeffs: Sequence[Any]) -> list[Any]:
    """Flatten a decomposition into one list, coarsest band first.

    Both our decomposition and PyWavelets' return `[approx, details, ...]`, so
    sorting each level's keys puts the two in the same order.

    Args:
        coeffs: Decomposition to flatten.
    """
    flat = [coeffs[0]]
    for level in coeffs[1:]:
        flat.extend(level[key] for key in sorted(level))
    return flat


def as_arrays(coeffs: Sequence[Any]) -> list:
    """Return a decomposition as numpy arrays, for comparison across backends.

    Args:
        coeffs: Decomposition to convert.
    """
    return [
        band.detach().cpu().numpy() if torch.is_tensor(band) else band
        for band in flatten(coeffs)
    ]


@dataclass
class Backend:
    """One implementation under test."""

    name: str
    decompose: Callable[[Any, Case], Any]
    to_input: Callable[[torch.Tensor, Case], Any]
    to_arrays: Callable[[Any], list] = None
    devices: tuple[str, ...] = ("cpu", "cuda")
    dimensions: tuple[int, ...] = (1, 2, 3)
    note: str = ""
    differentiable: bool = True

    def arrays(self, coeffs: Any) -> list:
        """Return a decomposition in the canonical order, as numpy arrays.

        Args:
            coeffs: Decomposition in whatever form the backend returned.
        """
        return (self.to_arrays or as_arrays)(coeffs)


def _torch_input(x: torch.Tensor, case: Case) -> torch.Tensor:
    """Return the input a torch backend takes."""
    return x


def _numpy_input(x: torch.Tensor, case: Case):
    """Return the input a numpy backend takes."""
    return x.detach().cpu().numpy()


def _core(x: torch.Tensor, case: Case):
    """Decompose with the pure-torch core."""
    return wavedecn(x, case.wavelet, case.mode, case.levels, case.axes)


def _kernels(x: torch.Tensor, case: Case):
    """Decompose with the compiled kernels."""
    from chuchichaestli.dwt import _ext

    return _ext.wavedecn(x, case.wavelet, case.mode, case.levels, case.axes)


def _pywt(x, case: Case):
    """Decompose with PyWavelets."""
    import pywt

    return pywt.wavedecn(x, case.wavelet, mode=case.mode, level=case.levels, axes=case.axes)


def _ptwt(x: torch.Tensor, case: Case):
    """Decompose with the PyTorch Wavelet Toolbox."""
    import ptwt

    entry = {1: ptwt.wavedec, 2: ptwt.wavedec2, 3: ptwt.wavedec3}[case.dimensions]
    return entry(x, case.wavelet, level=case.levels, mode=case.mode)


def _ptwt_arrays(coeffs: Sequence[Any]) -> list:
    """Put a PyTorch Wavelet Toolbox decomposition into the canonical order.

    Its one-dimensional levels are bare tensors and its two-dimensional ones are
    `(da, ad, dd)` tuples, where the canonical order is the sorted key order
    `(ad, da, dd)`; its three-dimensional levels are already keyed dicts.

    Args:
        coeffs: Decomposition to convert.
    """
    flat = [coeffs[0]]
    for level in coeffs[1:]:
        if torch.is_tensor(level):
            flat.append(level)
        elif isinstance(level, dict):
            flat.extend(level[key] for key in sorted(level))
        else:
            horizontal, vertical, diagonal = level
            flat.extend((vertical, horizontal, diagonal))
    return [band.detach().cpu().numpy() for band in flat]


def _pytorch_wavelets(x: torch.Tensor, case: Case):
    """Decompose with the pytorch_wavelets package."""
    import pytorch_wavelets

    transform = pytorch_wavelets.DWTForward(
        J=case.levels, wave=case.wavelet, mode=case.mode
    ).to(x.device)
    return transform(x)


def _pytorch_wavelets_arrays(coeffs: Any) -> list:
    """Put a pytorch_wavelets decomposition into the canonical order.

    It returns `(approx, details)` with the details finest first and their three
    bands stacked as `(da, ad, dd)` on their own axis.

    Args:
        coeffs: Decomposition to convert.
    """
    approx, details = coeffs
    flat = [approx]
    for level in reversed(details):
        flat.extend((level[:, :, 1], level[:, :, 0], level[:, :, 2]))
    return [band.detach().cpu().numpy() for band in flat]


BACKENDS: dict[str, Backend] = {
    "torch": Backend("torch", _core, _torch_input),
    "kernel": Backend("kernel", _kernels, _torch_input, note="compiled extension"),
    "pywt": Backend(
        "pywt",
        _pywt,
        _numpy_input,
        devices=("cpu",),
        note="single-threaded C on numpy, no autograd",
        differentiable=False,
    ),
    "ptwt": Backend(
        "ptwt",
        _ptwt,
        _torch_input,
        _ptwt_arrays,
        note="PyTorch Wavelet Toolbox",
    ),
    "ptwavelets": Backend(
        "ptwavelets",
        _pytorch_wavelets,
        _torch_input,
        _pytorch_wavelets_arrays,
        dimensions=(2,),
        note="pytorch_wavelets, 2d only, rebuilds its filters per call",
    ),
}


def probe(backend: Backend, device: str, case: Case) -> str:
    """Return `'ok'`, or why the backend cannot run this case.

    Args:
        backend: Backend to probe.
        device: Device to run on.
        case: Case to run.
    """
    if device not in backend.devices:
        return f"no {device}"
    if case.dimensions not in backend.dimensions:
        return f"no {case.dimensions}d"
    try:
        x = torch.zeros(2, 1, *case.shape, dtype=case.dtype)
        backend.decompose(backend.to_input(x, case), case)
    except ImportError:
        return "not installed"
    except Exception as exc:  # noqa: BLE001 - any failure disqualifies the backend
        reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return reason[:44]
    return "ok"


def peak_memory(call: Callable[[], Any], device: str) -> float:
    """Return the peak allocation of one call, in mebibytes.

    Reported on the GPU only: on the CPU the caching allocator keeps torch
    tensors off the Python heap that `tracemalloc` can see, so the number would
    flatter torch and mean something else for numpy.

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


def check(
    backend: Backend, device: str, case: Case, x: torch.Tensor, reference: list | None
) -> bool:
    """Check a backend against the reference decomposition.

    Args:
        backend: Backend to check.
        device: Device to run on.
        case: Case to run.
        x: The very input the reference was computed from.
        reference: Reference coefficients, or `None` to skip the check.
    """
    if reference is None:
        return True
    import numpy as np

    ours = backend.arrays(backend.decompose(backend.to_input(x.to(device), case), case))
    if len(ours) != len(reference):
        return False
    # detail coefficients sit near zero, where a relative tolerance buys nothing,
    # so the gate has to be loose enough for the rounding of the working dtype
    atol = 1e-10 if case.dtype is torch.float64 else 1e-4
    return all(
        a.shape == b.shape and np.allclose(a, b, rtol=1e-4, atol=atol)
        for a, b in zip(ours, reference, strict=True)
    )


def sample(case: Case) -> torch.Tensor:
    """Draw the input for a case, identically for every backend.

    Args:
        case: Case to draw for.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 1, *case.shape, dtype=case.dtype, generator=generator)


def total_energy(coeffs: Any) -> torch.Tensor:
    """Sum every tensor of a decomposition, whatever shape the structure has.

    Walks the structure rather than the canonical arrays, which are detached.

    Args:
        coeffs: Decomposition to sum.
    """
    if torch.is_tensor(coeffs):
        return coeffs.sum()
    values = coeffs.values() if isinstance(coeffs, dict) else coeffs
    return sum(total_energy(value) for value in values)


def timed_call(
    backend: Backend, payload: Any, case: Case, direction: str
) -> Callable[[], Any]:
    """Return the callable the timer runs.

    Args:
        backend: Backend to call.
        payload: Input in whatever form the backend takes.
        case: Case to run.
        direction: `'forward'` or `'backward'`.
    """
    if direction == "forward":
        return lambda: backend.decompose(payload, case)

    def backward() -> None:
        total_energy(backend.decompose(payload, case)).backward()

    return backward


def run(args: argparse.Namespace) -> list[Result]:
    """Run the sweep and return one result per backend and case.

    Args:
        args: Parsed command line arguments.
    """
    dtype = getattr(torch, args.dtype)
    cases = [
        Case(dimensions, wavelet, mode, levels, tuple(shape), dtype)
        for dimensions, shape in zip(args.dims, args.sizes, strict=True)
        for wavelet in args.wavelets
        for mode in args.modes
        for levels in args.levels
    ]
    results: list[Result] = []

    for case in cases:
        x = sample(case)
        reference = None
        if "pywt" in args.backends and probe(BACKENDS["pywt"], "cpu", case) == "ok":
            reference = BACKENDS["pywt"].arrays(_pywt(_numpy_input(x, case), case))

        for name in args.backends:
            backend = BACKENDS[name]
            status = probe(backend, args.device, case)
            if status != "ok":
                results.append(Result(name, args.device, case, status))
                continue
            if not check(backend, args.device, case, x, reference):
                results.append(Result(name, args.device, case, "MISMATCH"))
                continue
            if args.direction == "backward" and not backend.differentiable:
                results.append(Result(name, args.device, case, "n/a"))
                continue

            payload = backend.to_input(x.to(args.device), case)
            if args.direction == "backward":
                payload = payload.requires_grad_(True)
            call = timed_call(backend, payload, case, args.direction)

            peak = peak_memory(call, args.device)
            timer = benchmark.Timer(
                stmt="call()",
                globals={"call": call},
                num_threads=args.threads or torch.get_num_threads(),
                label=f"wavelet transform ({args.direction}, {args.device})",
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


def report(results: list[Result], args: argparse.Namespace) -> None:
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


def write(results: list[Result], args: argparse.Namespace) -> None:
    """Write the results to the requested files.

    Args:
        results: Results to write.
        args: Parsed command line arguments.
    """
    rows = [
        {
            "backend": r.backend,
            "device": r.device,
            "dimensions": r.case.dimensions,
            "wavelet": r.case.wavelet,
            "mode": r.case.mode,
            "levels": r.case.levels,
            "shape": "x".join(str(n) for n in r.case.shape),
            "status": r.status,
            "median_ms": r.median_ms,
            "iqr_ms": r.iqr_ms,
            "peak_mb": r.peak_mb,
        }
        for r in results
    ]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    if args.plot:
        plot(rows, args.plot)


def plot(rows: list[dict], path: str) -> None:
    """Draw the timings as a grouped bar chart.

    Args:
        rows: Result rows to draw.
        path: File to write the figure to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timed = [row for row in rows if row["status"] == "ok"]
    if not timed:
        print("nothing to plot")
        return
    cases = sorted({f"{r['dimensions']}d {r['wavelet']} {r['mode']}" for r in timed})
    backends = sorted({r["backend"] for r in timed})
    width = 0.8 / len(backends)
    figure, axes = plt.subplots(figsize=(2 + 1.4 * len(cases), 4))
    for i, backend in enumerate(backends):
        heights = [
            next(
                (
                    r["median_ms"]
                    for r in timed
                    if r["backend"] == backend
                    and f"{r['dimensions']}d {r['wavelet']} {r['mode']}" == case
                ),
                0.0,
            )
            for case in cases
        ]
        axes.bar([x + i * width for x in range(len(cases))], heights, width, label=backend)
    axes.set_xticks([x + 0.4 - width / 2 for x in range(len(cases))])
    axes.set_xticklabels(cases, rotation=30, ha="right")
    axes.set_ylabel("median time / ms")
    axes.set_yscale("log")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse; `sys.argv` if omitted.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--backends", nargs="+", default=["torch", "kernel", "pywt", "ptwt", "ptwavelets"])
    parser.add_argument("--dims", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=lambda s: [int(n) for n in s.split("x")],
        default=[[4096], [256, 256], [64, 64, 64]],
        help="one shape per entry of --dims, e.g. 256x256",
    )
    parser.add_argument("--wavelets", nargs="+", default=["haar", "db4", "db8"])
    parser.add_argument("--modes", nargs="+", default=["zero", "symmetric", "periodization"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--direction", default="forward", choices=("forward", "backward"))
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.5,
        help="seconds each measurement runs for; the timer picks the iteration count",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--plot", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark.

    Args:
        argv: Arguments to parse; `sys.argv` if omitted.
    """
    args = parse(argv)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no GPU available")
    results = run(args)
    report(results, args)
    write(results, args)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark the wavelet transform backends against each other and external libraries.

Every backend is checked against PyWavelets before it is timed, so a fast wrong
answer cannot win. Pull external libraries in for the run, and use one thread to
be fair to PyWavelets, which is single-threaded C on numpy:

    uv run --with pywavelets --with ptwt --with pytorch_wavelets python benches/dwt_impl.py --threads 1

Drop `--threads 1` for what the torch backends actually get. Further examples:

    # the torch libraries on the GPU (PyWavelets unable to run on GPU)
    python benches/dwt_impl.py --device cuda --backends c3li-torch ptwt ptwavelets

    # one case, backward, in double precision
    python benches/dwt_impl.py --dims 2 --sizes 512x512 --levels 3 --dtype float64 --direction backward

    # save the benchmark and plot it
    python benches/dwt_impl.py --json out.json --csv out.csv --plot out.png
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from chuchichaestli.benchmark import (
    Backend,
    Benchmark,
    TensorCase,
    base_parser,
    numpy_input,
    pin_allocator,
    shape,
    torch_input,
)
from chuchichaestli.dwt import _ext
from chuchichaestli.dwt.functional import wavedecn


@dataclass(frozen=True, kw_only=True)
class Case(TensorCase):
    """One point of the sweep: a wavelet, a mode and a depth, over one shape."""

    dimensions: int
    wavelet: str
    mode: str
    levels: int

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


def only_dims(*dimensions: int) -> Callable[[Case], str]:
    """Return a support test accepting only these dimensionalities.

    Args:
        dimensions: Dimensionalities the backend implements.
    """

    def supports(case: Case) -> str:
        return "" if case.dimensions in dimensions else f"no {case.dimensions}d"

    return supports


@contextlib.contextmanager
def using_kernels(enabled: bool):
    """Force the compiled kernels on or off for the duration of a call.

    Args:
        enabled: Whether the compiled kernels may serve the call.
    """
    previous = _ext.USE_CUSTOM_KERNELS
    _ext.USE_CUSTOM_KERNELS = enabled
    try:
        yield
    finally:
        _ext.USE_CUSTOM_KERNELS = previous


def _core(x: torch.Tensor, case: Case):
    """Decompose with the pure-torch core."""
    with using_kernels(False):
        return wavedecn(x, case.wavelet, case.mode, case.levels, case.axes)


def _kernels(x: torch.Tensor, case: Case):
    """Decompose through the compiled kernels.

    Raises:
        ImportError: If the extension was not built, so the backend reports
            itself missing rather than quietly timing the torch path again.
    """
    if not _ext.kernels_built():
        raise ImportError("the extension is not built")
    with using_kernels(True):
        return wavedecn(x, case.wavelet, case.mode, case.levels, case.axes)


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


# the `c3li-` prefix marks what this package provides, against the external
# libraries it is measured with
BACKENDS: dict[str, Backend] = {
    "c3li-torch": Backend("c3li-torch", _core, torch_input, as_arrays),
    "c3li-kernel": Backend(
        "c3li-kernel", _kernels, torch_input, as_arrays, note="compiled extension"
    ),
    "pywt": Backend(
        "pywt",
        _pywt,
        numpy_input,
        as_arrays,
        devices=("cpu",),
        note="single-threaded C on numpy, no autograd",
        differentiable=False,
    ),
    "ptwt": Backend(
        "ptwt",
        _ptwt,
        torch_input,
        _ptwt_arrays,
        note="PyTorch Wavelet Toolbox",
    ),
    "ptwavelets": Backend(
        "ptwavelets",
        _pytorch_wavelets,
        torch_input,
        _pytorch_wavelets_arrays,
        supports=only_dims(2),
        note="pytorch_wavelets, 2d only, rebuilds its filters per call",
    ),
}


def build_cases(args: argparse.Namespace) -> list[Case]:
    """Expand the command line into the cases of the sweep.

    Args:
        args: Parsed command line arguments.
    """
    dtype = getattr(torch, args.dtype)
    return [
        Case(
            shape=tuple(shape),
            dtype=dtype,
            dimensions=dimensions,
            wavelet=wavelet,
            mode=mode,
            levels=levels,
        )
        for dimensions, shape in zip(args.dims, args.sizes, strict=True)
        for wavelet in args.wavelets
        for mode in args.modes
        for levels in args.levels
    ]


def reference(case: Case, x: torch.Tensor, args: argparse.Namespace) -> list | None:
    """Return what every backend is checked against, which is PyWavelets.

    It is the outside reference, so it settles what the right answer is; when it
    is not in the run, or cannot express a case, nothing is checked.

    Args:
        case: Case to compute the reference for.
        x: The very input every backend will be given.
        args: Parsed command line arguments.
    """
    if "pywt" not in args.backends:
        return None
    if BACKENDS["pywt"].probe("cpu", case) != "ok":
        return None
    return BACKENDS["pywt"].arrays(_pywt(numpy_input(x, case), case))


def traffic(case: Case) -> float:
    """Bytes one decomposition has to move at least, read plus written.

    Every level reads its input once and writes as much again, and each level
    works on `2**-dimensions` of the one before it. Comparing this against the
    machine's memory bandwidth says how much of the ceiling a kernel reaches,
    which a timing on its own cannot.

    Args:
        case: Case to size.
    """
    return sum(
        2 * case.nbytes / (2 ** (case.dimensions * level)) for level in range(case.levels)
    )


def case_argv(case: Case) -> list[str]:
    """Return the flags selecting one case, for the `perf stat` worker.

    Args:
        case: Case to name.
    """
    return [
        "--dims", str(case.dimensions),
        "--sizes", "x".join(str(n) for n in case.shape),
        "--wavelets", case.wavelet,
        "--modes", case.mode,
        "--levels", str(case.levels),
    ]


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line: the shared arguments plus this sweep's axes.

    Args:
        argv: Arguments to parse; `sys.argv` if omitted.
    """
    parser = base_parser(__doc__, backends=list(BACKENDS))
    parser.add_argument("--dims", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=shape,
        default=[[4096], [256, 256], [64, 64, 64]],
        help="one shape per entry of --dims, e.g. 256x256",
    )
    parser.add_argument("--wavelets", nargs="+", default=["haar", "db4", "db8"])
    parser.add_argument("--modes", nargs="+", default=["zero", "symmetric", "periodization"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--direction", default="forward", choices=("forward", "backward"))
    return parser.parse_args(argv)


BENCHMARK = Benchmark(
    backends=BACKENDS,
    cases=build_cases,
    label="wavelet transform",
    reference=reference,
    group=lambda row: f"{row['dimensions']}d {row['wavelet']} {row['mode']}",
    case_argv=case_argv,
    moved_bytes=traffic,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark.

    Args:
        argv: Arguments to parse; `sys.argv` if omitted.
    """
    BENCHMARK.main(parse(argv))


if __name__ == "__main__":
    pin_allocator()
    main()

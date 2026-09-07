# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The command line every benchmark shares.

`base_parser` carries what any benchmark in chuchichaestli should include:
backends, device, measure time, result export, etc.

Extend it with benchmark-specific arguments as follows
```python
parser = base_parser(__doc__, backends=list(BACKENDS))
parser.add_argument("--specific-arg", nargs="+", default=["some-default"])
```
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence


def shape(text: str) -> list[int]:
    """Parse one `x`-separated shape, as in `2x3x256x256`.

    Args:
        text: Shape to parse.

    Raises:
        argparse.ArgumentTypeError: If a component is not a positive integer.
    """
    try:
        sizes = [int(n) for n in text.split("x")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an x-separated shape") from None
    if not sizes or any(n <= 0 for n in sizes):
        raise argparse.ArgumentTypeError(f"{text!r} has a non-positive extent")
    return sizes


def base_parser(
    description: str | None = None,
    backends: Sequence[str] = (),
    dtypes: Sequence[str] = ("float32", "float64"),
) -> argparse.ArgumentParser:
    """Return a parser holding the arguments common to every benchmark.

    The caller adds the axes of its own sweep and parses the result itself, so
    the two sets of arguments end up in one namespace and one `--help`.

    Args:
        description: Help text, usually the calling module's docstring.
        backends: Names to accept for `--backends`, in the order they run.
        dtypes: Dtype names to accept for `--dtype`; the first is the default.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--backends", nargs="+", default=list(backends), choices=list(backends))
    parser.add_argument("--dtype", default=dtypes[0], choices=tuple(dtypes))
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch intra-op threads; the torch default if omitted",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.5,
        help="seconds each measurement runs for; the timer picks the iteration count",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "measure the whole sweep this many times, each in a fresh process,"
            " and report the median of the medians"
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print where the time goes per operator, through torch.profiler",
    )
    parser.add_argument(
        "--trace",
        default=None,
        metavar="DIR",
        help=(
            "with --profile, also write a Chrome trace per case into this"
            " directory, for a timeline view at https://ui.perfetto.dev"
        ),
    )
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--profile-rows", type=int, default=10)
    parser.add_argument(
        "--perf",
        action="store_true",
        help="report hardware counters through `perf stat`, for the headroom left",
    )
    parser.add_argument("--perf-iterations", type=int, default=200)
    parser.add_argument("--perf-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--from-json",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "report and plot these saved result files instead of measuring;"
            " several are merged, so a sweep split one backend per process"
            " can be drawn as one figure"
        ),
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--plot", default=None)
    return parser


def without_options(argv: Sequence[str], options: Iterable[str]) -> list[str]:
    """Return a command line with these options, and the value of each, removed.

    Both spellings go, and a flag takes exactly one following token with it,
    whether or not that token looks like another flag.

    Args:
        argv: Arguments to filter, without the program name.
        options: Option names to remove.
    """
    options = set(options)
    kept: list[str] = []
    drop = False
    for a in argv:
        if drop:
            drop = False
            continue
        if a in options:
            drop = True
            continue
        if a.split("=")[0] in options:
            continue
        kept.append(a)
    return kept

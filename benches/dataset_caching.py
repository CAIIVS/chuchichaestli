# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark what stochastic caching buys a CachingDataset, fraction by fraction.

A `CachingDataset` keeps as many samples in shared memory as its cache size has
room for. When that is less than the whole dataset only the samples it fits are
kept, which a shuffled loader turns into a hit rate equal to the fraction
cached -- so a 2 GiB dataset with a 1 GiB cache serves half of every epoch from
memory. This sweeps that fraction and reports what an epoch then costs, against
the very same case with no cache at all.

The cache is filled before anything is timed, so every measurement is a
steady-state epoch rather than the one that paid for the cache; that first epoch
is what `benches/dataset_types.py` measures. The page cache is left alone, so a
reported speedup is what the sample cache earns over reads the operating system
was already serving from memory -- a lower bound, and the reproducible one.

    # the default sweep: no cache through to the whole dataset, per format
    python benches/dataset_caching.py

    # a 2 GiB dataset, uncached against half cached, over four workers
    python benches/dataset_caching.py --samples 8192 --fractions 0 0.5 --workers 4

    # saved, plotted, and reported again later without measuring anything
    python benches/dataset_caching.py --json caching.json --plot caching.png
    python benches/dataset_caching.py --from-json caching.json

A fraction is the share of the samples cached, and the cache is asked for what
holding them takes: the slots, and the byte per sample the cache spends on their
slot states. Every run prints what each cache actually took.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from chuchichaestli.benchmark import (
    Backend,
    Benchmark,
    base_parser,
    load_rows,
    pin_allocator,
    read_epoch,
    report,
    report_rows,
    sweep,
    write_rows,
)
from chuchichaestli.data import CachingDataset
from chuchichaestli.utils import nbytes
from _fixtures import (
    DATA_FORMATS,
    DataFormat,
    DatasetCase,
    add_data_options,
    build_datasets,
    data_case_kwargs,
    open_dataset,
)


@dataclass(frozen=True, kw_only=True)
class CachedCase(DatasetCase):
    """One dataset, read by a loader, with this share of it held in memory.

    Args:
        fraction: Share of the samples the cache is sized for.
    """

    fraction: float = 0.0

    @property
    def cache(self) -> nbytes:
        """Size the sample cache is asked for, its slot states included.

        A cache holds a byte per sample beside the slots, so a fraction that
        covered the samples alone would leave the states to come out of them,
        and every fraction would fall short of the share it names.
        """
        if not self.fraction:
            return nbytes(0)
        return nbytes(self.fraction * self.dataset_bytes + self.samples)

    def family(self) -> str:
        """The same case with the fraction left out, which a speedup is against."""
        return super().label()

    def label(self) -> str:
        """Short description, used as the row name in every report."""
        return f"{self.family()} c{self.fraction:.0%}"

    def fields(self) -> dict[str, Any]:
        """Every field of the case, plus the family a speedup is measured in."""
        return {**super().fields(), "family": self.family()}


_LAST: CachingDataset | None = None


def warm_dataset(fmt: DataFormat, x: torch.Tensor, case: CachedCase) -> tuple:
    """Open the dataset with its cache already filled, and say what it took.

    The one opened before is closed first: a cache the size of the dataset is
    too much shared memory to leave lying around for the garbage collector.

    Args:
        fmt: Format to read.
        x: Placeholder the sweep drew, whose device the batches are read onto.
        case: Case to read.
    """
    global _LAST
    if _LAST is not None:
        _LAST.close()
    _LAST = open_dataset(fmt, case, case.cache, preload=True)
    print(
        f"  {case.label():34s} {_LAST.n_cached:6d}/{case.samples} samples"
        f" cached in {_LAST.cache_size.as_str()}"
    )
    return _LAST, x.device


def cached_epoch(payload: tuple, case: CachedCase) -> None:
    """Walk one epoch of a dataset whose cache is already full.

    Args:
        payload: The open dataset and the device its batches are landed on.
        case: Case being read.
    """
    dataset, device = payload
    read_epoch(dataset, case, device)


BACKENDS: dict[str, Backend] = {
    name: Backend(name, cached_epoch, partial(warm_dataset, fmt), differentiable=False)
    for name, fmt in DATA_FORMATS.items()
}


def report_speedup(rows: Sequence[dict]) -> None:
    """Print what each fraction bought, against the same case with no cache.

    Args:
        rows: Result rows to summarise.
    """
    timed = [r for r in rows if r["status"] == "ok" and r.get("median_ms")
             and r.get("family")]
    baseline = {(r["backend"], r["family"]): r["median_ms"] for r in timed
                if float(r["fraction"]) == 0.0}
    if not timed:
        return
    if not baseline:
        print("\nnothing to compare against; put 0 in --fractions")
        return
    print("\nspeedup over the same case with no cache")
    for r in sorted(timed, key=lambda r: (r["family"], r["backend"],
                                          float(r["fraction"]))):
        base = baseline.get((r["backend"], r["family"]))
        ratio = f"{base / r['median_ms']:5.2f}x" if base else "     -"
        print(f"  {r['family']:30s} {r['backend']:12s}"
              f" {float(r['fraction']):5.0%} {r['median_ms']:9.2f} ms {ratio}")


class CachingBenchmark(Benchmark):
    """The shared driver, with the speedup summary this benchmark exists for.

    `Benchmark` reports what an epoch cost; what a cache fraction bought only
    shows against the same case uncached, which takes a second table.
    """

    def measure(self, args) -> None:
        """Time every backend over every case, then report and write.

        Args:
            args: Parsed command line arguments.
        """
        results = sweep(self.cases(args), self.backends, args, label=self.label)
        rows = [result.row() for result in results]
        report(results, args)
        report_speedup(rows)
        write_rows(rows, args, by_format(rows), self.series)

    def redraw(self, args) -> None:
        """Report and plot saved results, measuring nothing.

        Args:
            args: Parsed command line arguments.
        """
        rows = load_rows(args.from_json)
        report_rows(rows)
        report_speedup(rows)
        write_rows(rows, args, by_format(rows), self.series)


def build_cases(args: argparse.Namespace) -> list[CachedCase]:
    """Expand the command line into the cases of the sweep.

    Args:
        args: Parsed command line arguments.
    """
    return [CachedCase(**kwargs, fraction=share)
            for kwargs in data_case_kwargs(args)
            for share in args.fractions]


def by_format(rows: Sequence[dict]) -> Callable[[dict], str]:
    """Return what to draw a format's fractions together under.

    The format alone names the bars of a single dataset, and carries the case
    it was read in as well once a sweep holds more than one.

    Args:
        rows: Result rows to be drawn.
    """
    if len({row["family"] for row in rows}) < 2:
        return lambda row: row["backend"]
    return lambda row: f"{row['backend']} {row['family']}"


def by_fraction(row: dict) -> str:
    """Name one bar of a format, which is what a fraction bought.

    Args:
        row: Result row to name.
    """
    return f"c{float(row['fraction']):.0%}"


BENCHMARK = CachingBenchmark(
    backends=BACKENDS, cases=build_cases, label="cached dataset epoch",
    series=by_fraction,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark.

    `--repeats` measures in child processes and reports their median itself, so
    pass `--json` alongside it to have the speedup summary drawn from the rows.

    Args:
        argv: Arguments to parse; `sys.argv` if omitted.
    """
    parser = base_parser(__doc__, backends=list(BACKENDS))
    add_data_options(parser)
    parser.set_defaults(workers=[0], order=["shuffled"])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="share of the dataset the cache is sized for",
    )
    args = parser.parse_args(argv)
    if not args.from_json:
        build_datasets(build_cases(args), [DATA_FORMATS[n] for n in args.backends])
    BENCHMARK.main(args)
    if args.repeats > 1 and args.json:
        report_speedup(load_rows([args.json]))


if __name__ == "__main__":
    pin_allocator()
    main()

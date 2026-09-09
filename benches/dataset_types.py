# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark the various file formats supported in chuchichaestli.

Each backend is handed the very same samples written in its own format, and what
is measured is a `DataLoader` walking the dataset once, including opening the
files. Files are dropped from the page cache before every run so the reads go to
storage. The sample cache is off throughout, so this benchmark compares only
formats.

    # the default sweep: 1024 samples of 1x256x256, shuffled and ordered, 0 and 4 workers
    python benches/dataset_types.py

    # a 2 GiB dataset read shuffled through eight workers
    python benches/dataset_types.py --samples 8192 --workers 8 --order shuffled

    # what the batch costs, over one dataset
    python benches/dataset_types.py --batch-size 1 8 64

    # four runs each in a fresh process, then saved, plotted and redrawn
    python benches/dataset_types.py --repeats 4
    python benches/dataset_types.py --json types.json --plot types.png
    python benches/dataset_types.py --from-json types.json

The datasets are written under `--data-dir` once and reused; delete a file to
have it written again.

> Note: Keep the data directory on real storage, since nothing on `tmpfs` can be
> dropped from the page cache.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Sequence

import torch

from chuchichaestli.benchmark import (
    Backend,
    Benchmark,
    base_parser,
    drop_page_cache,
    pin_allocator,
    read_epoch,
)
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


def first_epoch(fmt: DataFormat, device: torch.device, case: DatasetCase) -> None:
    """Open the dataset cold and walk it once, which is what a first epoch is.

    Args:
        fmt: Format to read.
        device: Device the batches are landed on.
        case: Case being read.
    """
    drop_page_cache(case.path(fmt))
    dataset = open_dataset(fmt, case, nbytes(0))
    try:
        read_epoch(dataset, case, device)
    finally:
        dataset.close()


BACKENDS: dict[str, Backend] = {
    name: Backend(
        name,
        functools.partial(first_epoch, fmt),
        lambda x, case: x.device,
        differentiable=False,
    )
    for name, fmt in DATA_FORMATS.items()
}


def build_cases(args: argparse.Namespace) -> list[DatasetCase]:
    """Expand the command line into the cases of the sweep.

    Args:
        args: Parsed command line arguments.
    """
    return [DatasetCase(**kwargs) for kwargs in data_case_kwargs(args)]


BENCHMARK = Benchmark(
    backends=BACKENDS, cases=build_cases, label="dataset first epoch"
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark.

    Args:
        argv: Arguments to parse; `sys.argv` if omitted.
    """
    parser = base_parser(__doc__, backends=list(BACKENDS))
    add_data_options(parser)
    args = parser.parse_args(argv)
    if not args.from_json:
        build_datasets(build_cases(args), [DATA_FORMATS[n] for n in args.backends])
    BENCHMARK.main(args)


if __name__ == "__main__":
    pin_allocator()
    main()

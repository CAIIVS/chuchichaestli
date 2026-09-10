# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmarking utilities: a shared CLI, sweep runner, and summaries."""

from chuchichaestli.benchmark.args import base_parser, shape, without_options
from chuchichaestli.benchmark.benchmark import Benchmark, worker_command
from chuchichaestli.benchmark.data import DataLoaderCase, drop_page_cache, read_epoch
from chuchichaestli.benchmark.profile import perf_case, perf_counters, profile_case
from chuchichaestli.benchmark.utils import (
    ALLOCATOR_ENV,
    Backend,
    Case,
    Result,
    TensorCase,
    as_measurements,
    load_rows,
    natural,
    numpy_input,
    peak_memory,
    pin_allocator,
    plot,
    repeated,
    report,
    report_rows,
    sweep,
    torch_input,
    total_energy,
    write,
    write_rows,
)


__all__ = [
    "ALLOCATOR_ENV",
    "Backend",
    "Benchmark",
    "Case",
    "DataLoaderCase",
    "Result",
    "TensorCase",
    "as_measurements",
    "base_parser",
    "drop_page_cache",
    "load_rows",
    "natural",
    "numpy_input",
    "peak_memory",
    "perf_case",
    "perf_counters",
    "pin_allocator",
    "plot",
    "profile_case",
    "read_epoch",
    "repeated",
    "report",
    "report_rows",
    "shape",
    "sweep",
    "torch_input",
    "total_energy",
    "without_options",
    "worker_command",
    "write",
    "write_rows",
]

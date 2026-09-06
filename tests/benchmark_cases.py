# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases, backends and arguments the benchmark tests share.

Not a test module: `test_benchmark_utils` and `test_benchmark_harness` both
measure the same toy sweep, and it is only worth describing once.
"""

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from chuchichaestli.benchmark.utils import Backend, TensorCase, torch_input


@dataclass(frozen=True)
class Hand:
    """A case owing nothing to `TensorCase`, to keep the protocol structural."""

    side: int

    def label(self) -> str:
        """Name the case."""
        return f"{self.side}x{self.side}"

    def fields(self) -> dict[str, Any]:
        """Describe the case for the output files."""
        return {"side": self.side}

    def sample(self) -> torch.Tensor:
        """Draw the input."""
        return torch.ones(self.side)


@dataclass(frozen=True, kw_only=True)
class Sized(TensorCase):
    """A case on the shared base, varying one axis of its own."""

    kind: str = "square"

    def label(self) -> str:
        """Name the case."""
        return "x".join(str(n) for n in self.shape)


def square(side: int, **kwargs) -> Sized:
    """Return a square case of this many elements a side."""
    return Sized(shape=(side, side), **kwargs)


def halve(x: torch.Tensor, case: Sized) -> list[torch.Tensor]:
    """The work under test: a correct backend."""
    return [x * 0.5]


def double(x: torch.Tensor, case: Sized) -> list[torch.Tensor]:
    """The work under test: a backend that disagrees with the reference."""
    return [x * 2.0]


def reference(case: Sized, x: torch.Tensor) -> list:
    """What every backend is checked against."""
    return [(x * 0.5).numpy()]


def namespace(**overrides) -> argparse.Namespace:
    """Return the arguments a sweep needs, with the slow parts turned down."""
    args = dict(
        device="cpu",
        dtype="float32",
        threads=1,
        min_run_time=0.01,
        backends=["good"],
        json=None,
        csv=None,
        plot=None,
        repeats=1,
        profile=False,
        profile_repeats=1,
        profile_rows=3,
        trace=None,
        perf=False,
        perf_worker=False,
        perf_iterations=10,
        from_json=None,
    )
    return argparse.Namespace(**{**args, **overrides})


GOOD = Backend("good", halve, torch_input)
BAD = Backend("bad", double, torch_input)


WORKER = '''
import json, sys
rows = [{"backend": "good", "case": "8x8", "side": 8, "status": "ok",
         "median_ms": float(sys.argv[sys.argv.index("--scale") + 1])},
        {"backend": "absent", "case": "8x8", "side": 8, "status": "not installed",
         "median_ms": None}]
json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"))
'''

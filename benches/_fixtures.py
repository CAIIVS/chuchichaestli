# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The setup every benchmark reading a dataset off disk shares.

Not a benchmark: what the dataset sweeps have in common is the files they read,
and describing them once is what keeps two benchmarks comparable. A `DataFormat`
names one on-disk format, a `DatasetCase` one dataset and the loader walking it,
`build_datasets` writes the files, and `open_dataset` opens them checked.

```python
    parser = base_parser(__doc__, backends=list(BACKENDS))
    add_data_options(parser)
    args = parser.parse_args()
    cases = [DatasetCase(**kwargs) for kwargs in data_case_kwargs(args)]
    build_datasets(cases, [DATA_FORMATS[n] for n in args.backends])
```

Anything a further dataset benchmark needs to set itself up belongs here too.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import torch

from chuchichaestli.benchmark import TensorCase, shape
from chuchichaestli.data import (
    CachingDataset,
    HDF5Dataset,
    NumpyDataset,
    SafetensorsDataset,
    save_dataset,
)
from chuchichaestli.utils import nbytes


DATA_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "chuchichaestli"
    / "bench-data"
)


class DataFormat(NamedTuple):
    """One on-disk format: its extension, its dataset and that dataset's keys."""

    suffix: str
    dataset: type[CachingDataset]
    kwargs: Mapping[str, Any] = {}


DATA_FORMATS: dict[str, DataFormat] = {
    "hdf5": DataFormat(".h5", HDF5Dataset, {"groups": "data"}),
    "safetensors": DataFormat(".safetensors", SafetensorsDataset, {"keys": "data"}),
    "npy": DataFormat(".npy", NumpyDataset),
}


@dataclass(frozen=True, kw_only=True)
class DatasetCase(TensorCase):
    """One dataset on disk, and how a loader walks it.

    `shape` is one sample's shape, without the axis enumerating them, so
    `nbytes` is what a sample costs and `dataset_bytes` what an epoch delivers.

    Args:
        samples: Samples the dataset holds.
        data_dir: Directory the files are written under.
        batch_size: Samples one batch holds.
        workers: Loader worker processes; `0` reads in the main process.
        order: `'sequential'` or `'shuffled'`.
    """

    samples: int
    data_dir: str
    batch_size: int = 8
    workers: int = 0
    order: str = "sequential"

    # the sweep varies one sample's shape, not a batch of them
    lead: ClassVar[tuple[int, ...]] = ()

    @property
    def dataset_bytes(self) -> int:
        """Bytes one epoch delivers, which is the whole dataset."""
        return self.samples * self.nbytes

    def path(self, fmt: DataFormat) -> Path:
        """Return the file this dataset is written to in a format.

        Args:
            fmt: Format to locate.
        """
        stem = f"{self.samples}x{self.render(self.shape)}_{self.render(self.dtype)}"
        return Path(self.data_dir) / f"{stem}{fmt.suffix}"

    def sample(self) -> torch.Tensor:
        """Return a placeholder; the input of a dataset sweep is the file on disk.

        The sweep draws it only to hand `prepare` a tensor on its device.
        """
        return torch.zeros(1, dtype=self.dtype)

    def label(self) -> str:
        """Short description, used as the row name in every report."""
        return (
            f"{self.samples}x{super().label()} b{self.batch_size}"
            f" w{self.workers} {self.order[:3]}"
        )


def build_datasets(
    cases: Sequence[DatasetCase], formats: Sequence[DataFormat]
) -> None:
    """Write every file a sweep will read, before anything is measured.

    Each case's samples are drawn once and written to every format that is
    missing one, so the formats hold identical data and writing gigabytes is
    kept out of what the benchmark times.

    Args:
        cases: Cases of the sweep.
        formats: Formats every case is written in.
    """
    for case in cases:
        tensor = None
        for fmt in formats:
            path = case.path(fmt)
            if path.is_file():
                continue
            if tensor is None:
                tensor = torch.randn(
                    (case.samples, *case.shape),
                    dtype=case.dtype,
                    generator=torch.Generator().manual_seed(case.seed),
                )
            save_dataset(path, tensor)
            print(f"wrote {path} ({nbytes(path.stat().st_size).as_str()})")


def open_dataset(
    fmt: DataFormat, case: DatasetCase, cache: nbytes, preload: bool = False
) -> CachingDataset:
    """Open the dataset over a case's file, checked to hold what it should.

    Args:
        fmt: Format to read.
        case: Case to read.
        cache: Size of the shared-memory sample cache.
        preload: Whether to fill that cache before returning.

    Raises:
        ValueError: If the dataset does not hold what the case describes, so a
            backend that reads it wrongly is reported rather than timed.
    """
    dataset = fmt.dataset(
        str(case.path(fmt)),
        dtype=case.dtype,
        cache=cache,
        preload=preload,
        **fmt.kwargs,
    )
    held = (len(dataset), tuple(dataset.sample_shape))
    if held != (case.samples, case.shape):
        dataset.close()
        raise ValueError(f"holds {held}, expected {(case.samples, case.shape)}")
    return dataset


DATA_OPTIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--samples", {"nargs": "+", "type": int, "default": [1024]}),
    ("--sizes", {"nargs": "+", "type": shape, "default": [[1, 256, 256]],
                 "help": "shape of one sample, e.g. 1x256x256, sample axis aside"}),
    ("--workers", {"nargs": "+", "type": int, "default": [0, 4]}),
    ("--order", {"nargs": "+", "default": ["sequential", "shuffled"],
                 "choices": ("sequential", "shuffled"),
                 "help": "order the loader draws samples in"}),
    ("--batch-size", {"nargs": "+", "type": int, "default": [8]}),
    ("--data-dir", {"default": str(DATA_DIR),
                    "help": "where the datasets are written; real storage, not tmpfs"}),
)


def add_data_options(parser: argparse.ArgumentParser) -> None:
    """Add the arguments naming a dataset and the loader that walks it.

    A benchmark wanting other defaults follows this with `parser.set_defaults`.

    Args:
        parser: Parser to extend.
    """
    for flag, spec in DATA_OPTIONS:
        parser.add_argument(flag, **spec)


def data_case_kwargs(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Yield the constructor arguments of every dataset the command line names.

    A benchmark whose case carries more than these adds it to each of them.

    Args:
        args: Parsed command line arguments.
    """
    axes = product(
        args.sizes, args.samples, args.batch_size, args.workers, args.order
    )
    for extent, samples, batch_size, workers, order in axes:
        yield {
            "shape": tuple(extent), "dtype": getattr(torch, args.dtype),
            "samples": samples, "batch_size": batch_size,
            "workers": workers, "order": order, "data_dir": args.data_dir,
        }

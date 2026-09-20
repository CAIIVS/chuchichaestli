# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing a dataset that grows, one batch at a time."""

from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
import h5py
import torch
from chuchichaestli.data.hdf5 import HDF5Dataset
from chuchichaestli.data.save import save_dataset
from chuchichaestli.utils.io import staged
from chuchichaestli.utils.tensors import as_array


__all__ = ["Archive", "Hdf5Archive", "BufferedArchive", "ARCHIVES", "archive_for"]


class Archive(ABC):
    """A file written a batch at a time, landing atomically when closed.

    Attributes:
        path: File the batches land in.
        key: Name the batches are stored under, for formats that name.
        appends: Whether batches reach the file as they arrive, rather than
            being held until `close`.
    """

    appends: bool = False

    def __init__(self, path: str | Path, key: str = "data"):
        """Constructor.

        Args:
            path: File the batches land in.
            key: Name the batches are stored under, for formats that name.
        """
        self.path = Path(path)
        self.key = key

    def __repr__(self) -> str:
        """Return a short description of the archive."""
        return f"{type(self).__name__}({str(self.path)!r})"

    @abstractmethod
    def write(self, batch: torch.Tensor) -> None:
        """Add one batch of samples.

        Args:
            batch: Samples to add, the first axis being the batch.
        """

    @abstractmethod
    def close(self) -> None:
        """Finish the file and move it into place."""

    @abstractmethod
    def abort(self) -> None:
        """Give up, leaving nothing behind."""

    def __enter__(self) -> Archive:
        """Return the archive, for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close on success, abort on failure.

        Args:
            exc_type: Class of a raised exception, if one was raised.
            exc: The raised exception, if one was raised.
            traceback: Where it was raised.
        """
        if exc_type is None:
            self.close()
        else:
            self.abort()


class Hdf5Archive(Archive):
    """Appends each batch to a resizable HDF5 dataset."""

    appends = True

    def __init__(self, path: str | Path, key: str = "data"):
        """Constructor.

        Args:
            path: File the batches land in.
            key: Dataset the batches are written to.
        """
        super().__init__(path, key)
        self._stack = ExitStack()
        self._scratch = self._stack.enter_context(staged(self.path))[0]
        self._handle = h5py.File(self._scratch, "w")
        self._dataset: h5py.Dataset | None = None

    def write(self, batch: torch.Tensor) -> None:
        """Grow the dataset by one batch and fill the new rows.

        Args:
            batch: Samples to add, the first axis being the batch.
        """
        array = as_array(batch)
        if self._dataset is None:
            self._dataset = self._handle.create_dataset(
                self.key,
                shape=(0, *array.shape[1:]),
                maxshape=(None, *array.shape[1:]),
                dtype=array.dtype,
                chunks=True,
            )
        written = self._dataset.shape[0]
        self._dataset.resize(written + len(array), axis=0)
        self._dataset[written:] = array

    def close(self) -> None:
        """Close the file and move it into place."""
        self._handle.close()
        self._stack.close()

    def abort(self) -> None:
        """Close the file and discard it."""
        self._handle.close()
        self._stack.__exit__(RuntimeError, RuntimeError("aborted"), None)


class BufferedArchive(Archive):
    """Holds every batch until close, for formats that cannot be appended to.

    `.safetensors` and `.npz` fix their layout when the file is written, so
    there is nowhere to put a later batch.
    """

    def __init__(self, path: str | Path, key: str = "data"):
        """Constructor.

        Args:
            path: File the batches land in.
            key: Name the batches are stored under.
        """
        super().__init__(path, key)
        self._batches: list[torch.Tensor] = []

    def write(self, batch: torch.Tensor) -> None:
        """Keep one batch for the eventual write.

        Args:
            batch: Samples to add, the first axis being the batch.
        """
        self._batches.append(batch)

    def close(self) -> None:
        """Write everything held, in the order it arrived."""
        if self._batches:
            save_dataset(self.path, torch.cat(self._batches), key=self.key)
        self._batches = []

    def abort(self) -> None:
        """Drop everything held, writing nothing."""
        self._batches = []


ARCHIVES: dict[str, type[Archive]] = dict.fromkeys(
    HDF5Dataset.FILE_EXTENSIONS, Hdf5Archive
)


def archive_for(path: str | Path, key: str = "data", warn: bool = True) -> Archive:
    """Return an archive writing the format a path's suffix names.

    Args:
        path: File the batches land in.
        key: Name the batches are stored under.
        warn: Whether to say so when the format cannot be appended to, and
            every batch must therefore be held in memory.
    """
    path = Path(path)
    archive = ARCHIVES.get(path.suffix)
    if archive is not None:
        return archive(path, key)
    if warn:
        warnings.warn(
            f"{path.suffix!r} cannot be appended to, so every batch is held "
            f"until the file is written; use one of {sorted(ARCHIVES)} to "
            "write as you go.",
            stacklevel=2,
        )
    return BufferedArchive(path, key)

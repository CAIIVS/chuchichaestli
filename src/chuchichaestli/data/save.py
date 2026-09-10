# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing a dataset in any format chuchichaestli reads.

The datasets pick their reader by file extension; `save_dataset` is the
counterpart, so a fixture, a converted dataset, or the samples a benchmark
needs on disk are written the same way whatever the format:
```python
    save_dataset(path.with_suffix(".h5"), samples, attrs=metadata)
    HDF5Dataset(str(path.with_suffix(".h5")), groups="data", attrs_groups="attrs")
```

Metadata goes where the dataset reading it looks: a second group or dataset in
an HDF5 file, a second tensor in a safetensors file, a second array in a `.npz`
archive, and the sidecar `<stem>.attrs.npy` beside a `.npy` file. Every file is
written under a scratch name and moved into place, so an interrupted write
leaves nothing half written for the next read.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from safetensors.torch import save_file

from chuchichaestli.data.hdf5 import HDF5Dataset
from chuchichaestli.data.safetensors import SafetensorsDataset
from chuchichaestli.utils import as_array


__all__ = ["SAVERS", "save_dataset"]


@contextmanager
def _staged(*targets: Path) -> Iterator[list[Path]]:
    """Yield a scratch path per target; all are moved into place, or none are.

    Args:
        targets: Files the write is to land on.
    """
    scratches = [t.with_stem(f"{t.stem}.part") for t in targets]
    try:
        yield scratches
    except BaseException:
        for scratch in scratches:
            scratch.unlink(missing_ok=True)
        raise
    for scratch, target in zip(scratches, targets):
        scratch.replace(target)


def _as_tensor(data: Any) -> torch.Tensor:
    """Return the samples as a contiguous tensor, which is what safetensors takes.

    Args:
        data: Samples to write.
    """
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    return tensor.detach().cpu().contiguous()


def _attrs_array(attrs: Any, suffix: str, pickles: bool = False) -> np.ndarray:
    """Return metadata as an array the format can hold, one entry per sample.

    Args:
        attrs: Metadata to write.
        suffix: Suffix of the file being written, named in the errors.
        pickles: Whether the format holds what only pickling can, which of
            these formats the `.npy` sidecar alone does.

    Raises:
        ValueError: If the metadata is a mapping, which only HDF5 holds, or
            needs pickling to be stored in a format read back without it.
    """
    if isinstance(attrs, Mapping):
        raise ValueError(
            f"Cannot write a mapping to '{suffix}'; metadata of a whole"
            " dataset is HDF5 attributes, so write it to '.h5', or pass one"
            " entry per sample."
        )
    array = as_array(attrs)
    if array.dtype.kind == "O" and not pickles:
        raise ValueError(
            f"Cannot write metadata of dtype '{array.dtype}' to '{suffix}',"
            " which is read back without pickling; pass an array of numbers,"
            " or write to '.npy', whose sidecar holds anything picklable."
        )
    return array


def _sidecar(path: Path, attrs_key: str) -> Path:
    """Return the file a `.npy` dataset's metadata is read from beside it.

    Args:
        path: Dataset file the metadata belongs to.
        attrs_key: Name of the metadata, which is `NumpyDataset`'s
            `attrs_suffix` without its dot.
    """
    return path.with_name(f"{path.stem}.{attrs_key}{path.suffix}")


def _save_hdf5(path: Path, data: Any, key: str, attrs: Any, attrs_key: str) -> None:
    """Write the samples as one HDF5 dataset.

    A mapping is written as the HDF5 attributes of a group, anything else as a
    second dataset; `attrs_groups` reads either.

    Args:
        path: File to write.
        data: Samples to write.
        key: Group the samples are written under.
        attrs: Metadata to write, or `None`.
        attrs_key: Group or dataset the metadata is written under.
    """
    with _staged(path) as (scratch,):
        with h5py.File(scratch, "w") as handle:
            handle.create_dataset(key, data=as_array(data))
            if isinstance(attrs, Mapping):
                handle.create_group(attrs_key).attrs.update(attrs)
            elif attrs is not None:
                handle.create_dataset(
                    attrs_key, data=_attrs_array(attrs, path.suffix)
                )


def _save_safetensors(
    path: Path, data: Any, key: str, attrs: Any, attrs_key: str
) -> None:
    """Write the samples as one safetensors tensor.

    Metadata is a second tensor, not the file's `__metadata__` header, since
    that is what `attrs_keys` reads.

    Args:
        path: File to write.
        data: Samples to write.
        key: Key the samples are written under.
        attrs: Metadata to write, or `None`.
        attrs_key: Key the metadata is written under.
    """
    tensors = {key: _as_tensor(data)}
    if attrs is not None:
        tensors[attrs_key] = _as_tensor(_attrs_array(attrs, path.suffix))
    with _staged(path) as (scratch,):
        save_file(tensors, str(scratch))


def _save_npy(path: Path, data: Any, key: str, attrs: Any, attrs_key: str) -> None:
    """Write the samples as a single unnamed numpy array.

    A `.npy` file holds one array, so metadata goes to the sidecar beside it.

    Args:
        path: File to write.
        data: Samples to write.
        key: Unused; `.npy` holds one nameless array.
        attrs: Metadata to write, or `None`.
        attrs_key: Names the sidecar the metadata is written to.
    """
    targets = [path] if attrs is None else [path, _sidecar(path, attrs_key)]
    with _staged(*targets) as scratches:
        np.save(scratches[0], as_array(data))
        if attrs is not None:
            np.save(scratches[1], _attrs_array(attrs, path.suffix, pickles=True))


def _save_npz(path: Path, data: Any, key: str, attrs: Any, attrs_key: str) -> None:
    """Write the samples as one named array of an uncompressed archive.

    Args:
        path: File to write.
        data: Samples to write.
        key: Name the samples are written under.
        attrs: Metadata to write, or `None`.
        attrs_key: Name the metadata is written under.
    """
    arrays = {key: as_array(data)}
    if attrs is not None:
        arrays[attrs_key] = _attrs_array(attrs, path.suffix)
    with _staged(path) as (scratch,):
        np.savez(scratch, **arrays)


SAVERS: dict[str, Callable[[Path, Any, str, Any, str], None]] = {
    **dict.fromkeys(HDF5Dataset.FILE_EXTENSIONS, _save_hdf5),
    **dict.fromkeys(SafetensorsDataset.FILE_EXTENSIONS, _save_safetensors),
    ".npy": _save_npy,
    ".npz": _save_npz,
}


def save_dataset(
    path: str | Path,
    data: Any,
    key: str = "data",
    attrs: Any = None,
    attrs_key: str = "attrs",
) -> Path:
    """Write samples to a file, in the format its suffix names.

    Metadata enumerated by its first axis is read back one entry per sample,
    and every format holds numbers that way; only the `.npy` sidecar is
    pickled, so it alone holds an entry that is anything else, a dict per
    sample say. A mapping is metadata of the whole dataset instead, which
    only HDF5 holds, being the one format here with attributes of its own.

    Args:
        path: File to write; its suffix picks the format, and its directory is
            created if it does not exist.
        data: Samples to write, enumerated by the first axis.
        key: Name the samples are written under, for the formats that name
            them; it is what `HDF5Dataset(groups=...)` and
            `SafetensorsDataset(keys=...)` are then given to read them back.
        attrs: Metadata to write beside the samples, or `None` for a dataset
            without any; `attrs_groups` and `attrs_keys` read it back.
        attrs_key: Name the metadata is written under, which for `.npy` names
            the sidecar `<stem>.<attrs_key>.npy` instead, so it is
            `NumpyDataset`'s `attrs_suffix` without its dot.

    Raises:
        ValueError: If no format chuchichaestli reads has this suffix, or if
            the format cannot hold this metadata.
    """
    path = Path(path)
    saver = SAVERS.get(path.suffix)
    if saver is None:
        raise ValueError(
            f"Cannot write '{path.suffix}'; choose from {sorted(SAVERS)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    saver(path, data, key, attrs, attrs_key)
    return path

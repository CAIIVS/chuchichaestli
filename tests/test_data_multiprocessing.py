# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Picklability / multi-worker DataLoader tests for file datasets.

Guards `DataLoader(num_workers>0)` under both `fork` and `spawn`/`forkserver`
start methods (the latter require the dataset to be picklable).
"""

import pickle
import tempfile
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import h5py
import pytest
import torchvision.io as tvio
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from chuchichaestli.data.cache import nbytes
from chuchichaestli.data.numpy import NumpyDataset, ZipNumpyDataset
from chuchichaestli.data.hdf5 import HDF5Dataset
from chuchichaestli.data.safetensors import SafetensorsDataset
from chuchichaestli.data.image import ImageDataset

KINDS = ["numpy", "hdf5", "safetensors", "image", "zip"]
CONTEXTS = [c for c in ("fork", "spawn") if c in mp.get_all_start_methods()]


@pytest.fixture
def temp_dir():
    """A real on-disk temp dir (mmap-friendly, unlike pytest tmp_path here)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _first_tensor(x):
    """Extract the first tensor leaf from a tensor/tuple/list/dict sample."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        return _first_tensor(next(iter(x.values())))
    if isinstance(x, tuple | list):
        return _first_tensor(x[0])
    raise TypeError(f"no tensor leaf in {type(x)}")


def _build(kind, tmp_path):
    """Build a tiny dataset of the given kind; returns (dataset, n_samples)."""
    if kind == "numpy":
        files = []
        for i in range(3):
            p = tmp_path / f"n{i}.npy"
            np.save(p, np.random.randn(4, 3, 3).astype("float32"))
            files.append(str(p))
        return NumpyDataset(path=files, cache="4M"), 12
    if kind == "hdf5":
        fp = tmp_path / "a.h5"
        with h5py.File(fp, "w") as f:
            f.create_dataset("data/x", data=np.random.randn(10, 3, 3).astype("float32"))
        return HDF5Dataset(path=str(fp), groups="data/*", cache="4M"), 10
    if kind == "safetensors":
        fp = tmp_path / "s.safetensors"
        save_file({"data": torch.randn(10, 3, 3)}, str(fp))
        return SafetensorsDataset(path=str(fp), keys="*", cache="4M"), 10
    if kind == "image":
        files = []
        for i in range(3):
            t = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
            p = tmp_path / f"img{i}.jpg"
            tvio.write_jpeg(t, str(p))
            files.append(str(p))
        return ImageDataset(path=files, cache="4M"), 3
    if kind == "zip":
        a, b = [], []
        for i in range(3):
            pa, pb = tmp_path / f"a{i}.npy", tmp_path / f"b{i}.npy"
            np.save(pa, np.random.randn(4, 3, 3).astype("float32"))
            np.save(pb, np.random.randn(4, 3, 3).astype("float32"))
            a.append(str(pa))
            b.append(str(pb))
        return ZipNumpyDataset.from_named_paths({"a": a, "b": b}, cache="4M"), 12
    raise ValueError(kind)


def test_nbytes_pickle_roundtrip():
    """Pickle round-trip preserves an nbytes value (was blocked by units slot)."""
    for v in ("4G", "512M", 0):
        b = nbytes(v)
        r = pickle.loads(pickle.dumps(b))
        assert isinstance(r, nbytes) and float(r) == float(b)


class TestPickleRoundTrip:
    """Datasets round-trip through pickle and still serve samples."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_roundtrip(self, kind, temp_dir):
        """Unpickled dataset has the same length, index, and sample values."""
        ds, n = _build(kind, temp_dir)
        try:
            restored = pickle.loads(pickle.dumps(ds))
            assert len(restored) == len(ds) == n
            for i in (0, n - 1):
                assert torch.equal(_first_tensor(restored[i]), _first_tensor(ds[i]))
            restored.close()
        finally:
            ds.close()


class TestDataLoaderWorkers:
    """Full iteration under num_workers=2 for each available start method."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("ctx", CONTEXTS)
    def test_num_workers(self, kind, ctx, temp_dir):
        """Every sample is delivered exactly once by the worker processes."""
        ds, n = _build(kind, temp_dir)
        try:
            loader = DataLoader(
                ds, batch_size=2, num_workers=2, multiprocessing_context=ctx
            )
            seen = sum(_first_tensor(batch).shape[0] for batch in loader)
            assert seen == n
        finally:
            ds.close()


class TestSharedCache:
    """The shared-memory cache is populated by, and shared across, workers."""

    @pytest.mark.parametrize("ctx", CONTEXTS)
    def test_workers_populate_main_cache(self, ctx, temp_dir):
        """After a worker epoch the main process sees the cached slots.

        Under both fork and spawn/forkserver the workers write into the same
        segment the main process holds (attached by name under spawn, inherited
        under fork) -- not a per-worker copy.
        """
        ds, n = _build("numpy", temp_dir)  # cache="4M" holds all samples
        try:
            assert ds.cache.cached_states == 0
            loader = DataLoader(
                ds, batch_size=2, num_workers=2, multiprocessing_context=ctx
            )
            for _ in loader:
                pass
            assert ds.cache.cached_states > 0
        finally:
            ds.close()

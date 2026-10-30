# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic randomness: seeds derived from a root seed and a named position.

A resumed run recomputes its streams rather than replaying them. Ambient global
RNG is captured too, for code that ignores the generator it was handed.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


__all__ = [
    "root_seed",
    "draw_seed",
    "seeded_rand",
    "derive_seed",
    "rng_generator",
    "seed_ambient",
    "scoped_seed",
    "capture_rng_state",
    "restore_rng_state",
    "WorkerSeeder",
]

_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF

_SPARE_NUMPY: list[tuple[np.random.MT19937, np.random.RandomState]] = []


def root_seed(seed: int | None = None) -> int:
    """Return the seed given, or the one torch was last seeded with.

    Args:
        seed: An explicit seed, or `None` for `torch.initial_seed()`.
    """
    return torch.initial_seed() if seed is None else seed


def draw_seed(seed: int | None = None) -> int:
    """Return the seed given, or a fresh one drawn from torch's global stream.

    Args:
        seed: An explicit seed, or `None` to draw one.
    """
    if seed is not None:
        return seed
    return int(torch.empty((), dtype=torch.int64).random_())


def seeded_rand(
    *shape: int,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return uniform noise drawn from its own seed.

    Args:
        shape: Size of the tensor.
        seed: Seed the noise is drawn from.
        device: Device to put it on.
        dtype: Type to give it.
    """
    noise = torch.rand(*shape, generator=torch.Generator().manual_seed(seed))
    return noise.to(device=device, dtype=dtype)


def derive_seed(root: int, key: str) -> int:
    """Derive a stable seed for a named position.

    Uses a keyed BLAKE2b rather than the built-in `hash`, whose string hashing
    is salted per process and so would differ between a run and its resume.

    Args:
        root: Seed of the run as a whole.
        key: Names the position the seed is wanted for, e.g.
            `"program/0:train/data/epoch=3"`.
    """
    digest = hashlib.blake2b(
        key.encode("utf-8"),
        digest_size=8,
        key=(root & _MASK64).to_bytes(8, "little"),
    ).digest()
    return int.from_bytes(digest, "little")


def rng_generator(
    root: int, key: str, device: torch.device | str | None = None
) -> torch.Generator:
    """Build a generator seeded for a named position.

    Args:
        root: Seed of the run as a whole.
        key: Names the position the generator is wanted for.
        device: Device the generator draws for; defaults to the CPU.
    """
    gen = torch.Generator(device=device or "cpu")
    gen.manual_seed(derive_seed(root, key))
    return gen


def seed_ambient(root: int, key: str = "ambient") -> int:
    """Seed the global torch, numpy and python generators from a position.

    Args:
        root: Seed of the run as a whole.
        key: Names the position to derive the ambient seed from.

    Returns:
        The derived seed that was applied.
    """
    seed = derive_seed(root, key)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed & _MASK32)
    return seed


@contextmanager
def scoped_seed(seed: int) -> Iterator[None]:
    """Seed torch's CPU, numpy's and python's global generators for a block only.

    Args:
        seed: Seed the block draws from.
    """
    spare = _SPARE_NUMPY.pop() if _SPARE_NUMPY else _spare_numpy()
    torch_state = torch.get_rng_state()
    numpy_bits = np.random.get_bit_generator()
    python_state = random.getstate()
    torch.default_generator.manual_seed(seed)
    spare[1].seed(seed & _MASK32)
    np.random.set_bit_generator(spare[0])
    random.seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_bit_generator(numpy_bits)
        random.setstate(python_state)
        _SPARE_NUMPY.append(spare)


def _spare_numpy() -> tuple[np.random.MT19937, np.random.RandomState]:
    """Return a numpy generator and the legacy handle that reseeds it."""
    bits = np.random.MT19937()
    return bits, np.random.RandomState(bits)


def capture_rng_state() -> dict[str, Any]:
    """Capture the ambient torch, CUDA, MPS, numpy and python RNG state.

    Returns a flat mapping whose values are tensors or JSON-serializable
    scalars, so a checkpoint can route each to the right payload.
    """
    py_version, py_keys, py_gauss = random.getstate()
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python/version": py_version,
        "python/keys": torch.tensor(list(py_keys), dtype=torch.int64),
        "python/gauss": py_gauss,
        "numpy/kind": str(np_state[0]),
        "numpy/keys": torch.tensor(np_state[1].astype(np.int64), dtype=torch.int64),
        "numpy/pos": int(np_state[2]),
        "numpy/has_gauss": int(np_state[3]),
        "numpy/cached_gauss": float(np_state[4]),
    }
    if torch.cuda.is_available():
        for i, cuda_state in enumerate(torch.cuda.get_rng_state_all()):
            state[f"cuda/{i}"] = cuda_state
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore ambient RNG state captured by `capture_rng_state`.

    Missing entries are left alone, so a checkpoint taken on a CUDA or MPS host
    can be resumed on a CPU-only one.

    Args:
        state: Mapping as returned by `capture_rng_state`.
    """
    if "torch" in state:
        torch.set_rng_state(state["torch"].to(dtype=torch.uint8, device="cpu"))
    if "python/keys" in state:
        keys = tuple(int(k) for k in state["python/keys"].tolist())
        random.setstate(
            (int(state.get("python/version", 3)), keys, state.get("python/gauss"))
        )
    if "numpy/keys" in state:
        np.random.set_state(
            (
                state.get("numpy/kind", "MT19937"),
                state["numpy/keys"].to(torch.int64).numpy().astype(np.uint32),
                int(state.get("numpy/pos", 624)),
                int(state.get("numpy/has_gauss", 0)),
                float(state.get("numpy/cached_gauss", 0.0)),
            )
        )
    if torch.cuda.is_available():
        cuda_states = [state.get(f"cuda/{i}") for i in range(torch.cuda.device_count())]
        if all(s is not None for s in cuda_states):
            torch.cuda.set_rng_state_all(
                [s.to(dtype=torch.uint8, device="cpu") for s in cuda_states]
            )
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].to(dtype=torch.uint8, device="cpu"))


@dataclass(frozen=True, slots=True)
class WorkerSeeder:
    """A `worker_init_fn` deriving each dataloader worker's seed.

    Defined as a class rather than a closure so that it survives pickling into
    `spawn` and `forkserver` workers.

    Attributes:
        root: Seed of the run as a whole.
        key: Names the stream the workers serve.
    """

    root: int
    key: str

    def __call__(self, worker_id: int) -> None:
        """Seed the calling worker process.

        Args:
            worker_id: Index of the worker within the dataloader.
        """
        seed_ambient(self.root, f"{self.key}/worker={worker_id}")

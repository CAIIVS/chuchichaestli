# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: procedurally generated datasets.

This script shows how to construct, inspect, iterate, persist, and reload
various built-in `ProceduralDataset` subclasses:

* `HalfMoonsDataset`         - two interleaving half-moon shapes
* `SpiralsDataset`           - two interlocking Archimedean spirals
* `ConcentricSpheresDataset` - inner vs. outer sphere
* `GaussiansDataset`         - ring of Gaussian blobs
* `SwissRollDataset`         - spiral manifold in embedded in 3D

All datasets generate data in pure PyTorch, store it in shared
memory, and are fully compatible with `torch.utils.data.DataLoader`.

Run from repository root:
```
[uv run] python examples/procedural_datasets.py
```
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from chuchichaestli.data import (
    HalfMoonsDataset,
    SpiralsDataset,
    ConcentricSpheresDataset,
    GaussiansDataset,
    SwissRollDataset,
)
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt

# generate 1000 samples split into two half moons in 2D space
# preload the dataset into shared-memory cache before use
moons = HalfMoonsDataset(
    n_samples=1_000,
    noise=0.05,
    dim=2,
    seed=42,
    return_as="tuple",
    preload=True,
)
print(f"{moons}, shape={moons.shape}")
print(f"@index=0: sample shape={moons[0][0].shape}, label={moons[0][1]}\n")

# generate 1000 samples split into two Archimedean spiral arms in 2D space
spirals = SpiralsDataset(
    n_samples=1_000,
    noise=0.05,
    dim=2,
    seed=42,
    return_as="tuple",
)
print(f"{spirals}, shape={spirals.shape}")
print(f"@index=0: sample shape={spirals[0][0].shape}, label={spirals[0][1]}\n")

# generate 1000 samples split into two concentric spheres/rings in 2D space
spheres = ConcentricSpheresDataset(
    n_samples=1_000, noise=0.05, dim=2, seed=42, return_as="tuple"
)
print(f"{spheres}, shape={spheres.shape}")
print(f"@index=0: sample shape={spheres[0][0].shape}, label={spheres[0][1]}\n")

# generate 1000 samples split into 6 Gaussian blobs circularly arranged
gaussians = GaussiansDataset(
    n_samples=1_000, noise=0.05, dim=2, seed=42, return_as="tuple"
)
print(f"{gaussians}, shape={gaussians.shape}")
print(f"@index=0: sample shape={gaussians[0][0].shape}, label={gaussians[0][1]}\n")

# generate 1000 samples on a cylindrical spiral sheet embedded in 3D space
# preload the dataset into shared-memory cache before use
swissroll = SwissRollDataset(
    n_samples=1_000,
    noise=0.05,
    dim=3,
    seed=42,
    return_as="tuple",
    preload=True,
)
print(f"{swissroll}, shape={swissroll.shape}")
print(f"@index=0: sample shape={swissroll[0][0].shape}, label={swissroll[0][1]}\n")

# generate 1000 samples split into two concentric spheres in 2D space
spheres3d = ConcentricSpheresDataset(
    n_samples=1_000, noise=0.05, dim=3, seed=42, return_as="tuple"
)
print(f"{spheres3d}, shape={spheres3d.shape}")
print(f"@index=0: sample shape={spheres3d[0][0].shape}, label={spheres3d[0][1]}\n")

# draw each 2D dataset on a grid panel as scatter plots
fig = plt.figure(figsize=(11, 8))
for i, ds in enumerate([moons, spirals, spheres, gaussians]):
    ax = fig.add_subplot(2, 3, i + 1)
    # slicing works for ProceduralDataset subclasses (unless __getitem__ is overidden)
    samples, labels = ds[:]
    # colors correspond to the class labels
    colors = [list(TABLEAU_COLORS.values())[lbl] for lbl in labels.long()]
    ax.scatter(samples[:, 0], samples[:, 1], c=colors, s=6, alpha=0.75, lw=0)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax.set_title(ds.__class__.__name__)
# draw each 3D dataset on in a 3D volume as scatter plots
for i, ds3d in enumerate([swissroll, spheres3d], start=5):
    ax = fig.add_subplot(2, 3, i, projection="3d")
    samples, labels = ds3d[:]
    # for Swissroll, the labels are actually continuous,
    # but are collapsed to a single class for simplicity here
    colors = [list(TABLEAU_COLORS.values())[lbl] for lbl in labels.long()]
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=colors, s=6, alpha=0.75, lw=0)
    ax.view_init(elev=5, azim=80)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set_title(ds3d.__class__.__name__ + " (3D)")
plt.show()

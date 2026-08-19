# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: multi-field 2D DataLoader for the TNG50-1 projection dataset.

Each HDF5 file is named `<date>_tng50-1.<snapshot>.2D.<field>.<start>.<end>.hdf5`
and holds a `<field>/images` stack of shape `(4000, 512, 512)` (2D projection
maps) plus a `<field>/metadata` group with one per-image subgroup carrying HDF5
attributes (`gid`, `name`, `snapshot`, `extent`, ...). A `(snapshot, field)` pair
is sharded into three index ranges (`0000.1000`, `1000.2000`, `2000.3000`) that
concatenate into one sample axis; the six snapshots (50, 67, 78, 84, 91, 99)
span redshift.

This example pairs the `gas`, `dm`, and `star` fields into dict samples via
`ZipHDF5Dataset`, fetching each image's per-sample metadata through
`attrs_groups`, then iterates metadata-carrying batches from a plain
`DataLoader`. The final section visualizes one sample per field.

Run from repository root:
```
[uv run] python examples/tng50-1_2D_dataset.py
```

> Note: for this example to work, you need to set the `data_dir` to a directory
>       where the non-public TNG50-1.2D dataset is stored.
"""

import torch
from torch.utils.data import DataLoader, default_collate
from chuchichaestli.data import (
    ZipHDF5Dataset,
    ChannelExpand,
)
from chuchichaestli.data.cache import nbytes
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/scratch/data/illustris/tng50-1.2D"
# data_dir = "/raid/data/illustris/tng50-1.2D"

BATCH_SIZE = 8
SEED = 0x5EED
FIELDS = ["gas", "dm", "star"]
torch.manual_seed(SEED)

# `ZipHDF5Dataset` pairs the fields into dict samples. Each field's file shards
# are concatenate along the sample axis; `groups="*/images"` selects the single
# `<field>/images` cube per file, and `attrs_groups=*/metadata/*` selects the
# per-image metadata subgroups so each sample is `(image, metadata)`:
# - strict=True -> all fields must have the same number of samples.
zipds = ZipHDF5Dataset.from_named_paths(
    {f: f"{data_dir}/*tng50-1.*.2D.{f}.*.hdf5" for f in FIELDS},
    groups="*/images",
    attrs_groups="*/metadata/*",
    cache="512M",
    strict=True,
)
for ds in zipds.datasets:
    ds.info()

# A plain DataLoader shuffling across all snapshots and shards.
channel_expand = ChannelExpand()
loader = DataLoader(
    zipds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda s: {
        k: (channel_expand(v[0]), v[1]) for k, v in default_collate(s).items()
    },
    num_workers=0,
)
print("\nMetadata-carrying batches")
first_batch = None
for i, batch in enumerate(loader):
    # With attrs, each field is a `(image, metadata)` pair.
    images = {f: batch[f][0] for f in FIELDS}
    meta = {f: batch[f][1] for f in FIELDS}
    if first_batch is None:
        first_batch = batch
    print(", ".join(f"{f} {tuple(images[f].shape)}" for f in FIELDS))
    assert all(images[f].shape[:2] == (BATCH_SIZE, 1) for f in FIELDS)
    total = nbytes(sum(t.element_size() * t.nelement() for t in images.values()))
    print(f"  batch size: {total.to('M'):.1f} MiB")
    # per-image metadata (from the gas field)
    print("  (name, num_particles):")
    for m in [meta[f] for f in FIELDS]:
        for j in range(BATCH_SIZE):
            print(f"    name={m['name'][j]}, num_particles={m['num_particles'][j]}")
        print("---")
    if i == 1:
        break

# Visualize one sample across the three fields (from the first batch)
print("\nPlotting sample 0 of the first batch")
cmaps = {"gas": "inferno", "dm": "cividis", "star": "afmhot"}
gid = int(first_batch["gas"][1]["gid"][0])
fig, axs = plt.subplots(figsize=(15, 5), ncols=len(FIELDS), nrows=1)
for ax, f in zip(axs, FIELDS):
    img = np.log10(1 + first_batch[f][0][0, 0].numpy())  # (B, 1, H, W) -> (H, W)
    im = ax.imshow(img, cmap=cmaps[f], origin="lower")
    plt.colorbar(im, ax=ax, label=f"log$_{{10}}$ M$_\\mathrm{{{f}}}$")
    ax.set_title(f"{f} (gid {gid})")
plt.show()

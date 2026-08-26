# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: sequence-batched DataLoader for the ARTS4SKA dataset.

Each `.npy` volume is one integration step of a reionization simulation. The
redshift-label (e.g. `z9.940`) identifies a full convergence loop; the trailing
integer `N = 1, 2, ...` orders the sub-steps within a loop.
`xfrac` and `IonRates` are produced as a 1:1 pair per step.

This example builds a `DataLoader` whose *batch is one full convergence loop*,
i.e. the ordered sequence of paired `(xfrac, IonRates)` volumes, of variable
length across loops, via `HierarchicalFileBatchSampler.from_sequences` and
`sequence_collate`. A shared `RandomCropND` crops every step and paired field
of a loop with the same box.

Run from repository root:
```
[uv run] python examples/arts4ska_dataset.py
```

> Note: for this example to work, you need to set the `data_dir` to a directory
>       where the non-public ARTS4SKA dataset is stored.
"""

import re
import torch
from torch.utils.data import DataLoader
from chuchichaestli.data import (
    ZipNumpyDataset,
    HierarchicalFileBatchSampler,
    RandomCropND,
    sequence_collate,
    with_indices,
)
import matplotlib.pyplot as plt

data_dir = "/scratch/data/pasc/200Mpc_Nion_4_Source1_SinkA_pasc-ai"
# data_dir = "/raid/data/pasc/200Mpc_Nion_4_Source1_SinkA_pasc-ai"

SEED = 0x5EED
torch.manual_seed(SEED)

# Convergence-step files are named `<field>_z<label>_<N>.npy`
SEQ_PATTERN = r"_z([0-9.]+)_(\d+)"
rx = re.compile(SEQ_PATTERN)

# `ZipNumpyDataset` pairs xfrac & ionrates into dict samples. Wildcard paths
# are globbed and sorted internally (for str):
# - sample_axis=None -> each file is one 3D sample.
# - strict=True   -> pairing is 1:1, a count mismatch should raise.
zipds = ZipNumpyDataset.from_named_paths(
    {
        "xfrac": f"{data_dir}/xfrac_z*_[0-9]*.npy",
        "ion_rates": f"{data_dir}/IonRates_z*_[0-9]*.npy"
    },
    sample_axis=None,
    cache="512M",
    strict=True,
)
zipds.datasets[0].info()
zipds.datasets[1].info()

# One variable-length, step-ordered batch per redshift (convergence loop)
# The sampler is built on datasets[0] (xfrac); in practice, assert for a
# few indices that ZipDataset applies the same index to every subset.
# order_cast=int -> steps ordered low-N -> high-N within each loop;
# shuffle=True   -> randomize the loop (batch) order each epoch.
sampler = HierarchicalFileBatchSampler.from_sequences(
    zipds.datasets[0],
    pattern=SEQ_PATTERN,
    group=1,
    order_group=2,
    order_cast=int,
    shuffle=True,
)
print(f"Number of convergence loops (batches): {len(sampler)}")


# `RandomCropND` draws a crop box per batch and `sequence_collate` applies it
# identically to the batch and both fields. key_fn ensures filename provenance.
crop_collate = sequence_collate(
    transform=RandomCropND((128, 128, 128)),
    source=zipds.datasets,
    key_fn=lambda p: (m.group(1) if (m := rx.search(p.name)) else "__ungrouped__"),
)

# `with_indices` makes each sample carry its dataset index so the collate can
# attach provenance (key/files/indices).
loader = DataLoader(
    with_indices(zipds),
    batch_sampler=sampler,
    collate_fn=crop_collate,
    num_workers=0,
)

print("\nCropped, index-traceable batches")
for i, batch in enumerate(loader):
    print(
        f"z{batch['key']}: xfrac {tuple(batch['xfrac'].shape)}, "
        f"IonRates {tuple(batch['ion_rates'].shape)}"
    )
    assert batch["xfrac"].shape[0] == batch["ion_rates"].shape[0]
    # calculate total batch byte size (uncropped)
    batch_samples = batch["xfrac"].shape[0]
    uncropped = batch_samples * sum(ds.sample_size for ds in zipds.datasets)
    print(f"  total batch size: {uncropped.to('M'):.1f} MiB")
    # `files` is one step-ordered path list per subset (xfrac, ion_rates)
    print("  (step index: xfrac source, IonRates source):")
    for idx, xfrac_f, ion_rates_f in zip(batch["indices"], *batch["files"]):
        print(f"    {idx}: {xfrac_f.name}, {ion_rates_f.name}")
    if i == 1:
        break


# Same loader, but no transform and plain dict batches
print("\nPlotting uncropped, plain batch samples")
plain_loader = DataLoader(zipds, batch_sampler=sampler, num_workers=0)
first_batch = next(iter(plain_loader))
xf = first_batch["xfrac"][-1]  # (D, H, W)
ionr = first_batch["ion_rates"][-1]
mid = xf.shape[0] // 2

# Visualize mid-slices from the last volumes of the batch
fig, axs = plt.subplots(figsize=(12, 5), ncols=2, nrows=1)
im_xf = axs[0].imshow(xf[mid, :, :], cmap="seismic", origin="lower")
plt.colorbar(im_xf, ax=axs[0])
axs[0].set_title("xfrac (batch 0, step -1)")
im_ionr = axs[1].imshow(
    ionr[mid, :, :], cmap="PuOr", vmax=0.1 * float(ionr[mid, :, :].max()), origin="lower"
)
plt.colorbar(im_ionr, ax=axs[1])
axs[1].set_title("IonRates (batch 0, step -1)")
plt.show()

# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: loading HDF5 data with HDF5Dataset.

This script demonstrates the basic usage of `HDF5Dataset` for reading image data
from HDF5 files. The test data ./data/mnist_h5_tests/test_scenario_13 stores
digit images across multiple files in a `train/` `test/` and `val/` directory,
each file containing one or more image datasets under the `/image` group.

Run from repository root:
```
[uv run] python examples/hdf5_dataset.py
```
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from chuchichaestli.data import HDF5Dataset
import matplotlib.pyplot as plt

# select the training data in train
data_dir = Path("data/mnist_h5_tests/test_scenario_13/train")
# select all HDF5 files starting with 'mnist_test_' and ending with '.h5'
file_glob = str(data_dir / "mnist_test_*.h5")

# initialize the dataset
ds = HDF5Dataset(
    # the HDF5 files (wildcards * and ** work)
    path=file_glob,
    # the HDF5 groups (wildcards * and ** work)
    groups="image/*",
    # data type is casted when accessed
    dtype=torch.float64,
    # allocate 16M as cache (for larger datasets, the rest is read from disk)
    cache="16M",
)

# Print summary info about the dataset
ds.info()

# Create a data loader with the dataset
batch_size = 16
shuffle = True
num_workers = 0
print(f"Creating data loader with {batch_size=}, {shuffle=}, {num_workers=}")
loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
# Iterate through the dataset and plot the batches in a row
for step, batch in enumerate(loader):
    print(f"Batch {step:02d}, shape={batch.shape}, dtype={batch.dtype}")
    fig, axs = plt.subplots(ncols=len(batch), squeeze=False)
    for i, img in enumerate(batch):
        img = img.detach()
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
print()

# After the first iteration through the dataset the items are cached
print("After first epoch...")
ds.info()
ds.close()

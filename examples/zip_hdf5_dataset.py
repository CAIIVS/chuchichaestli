# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: loading HDF5 data with ZipHDF5Dataset.

This script shows how `ZipHDF5Dataset` lets you read multiple HDF5 groups/files,
so that each `__getitem__` call returns matching samples from every source
simultaneously.

The classmethod `ZipHDF5Dataset.from_groups` allows reading multiple HDF5 groups
from the same file sources, while `ZipHDF5Dataset.from_paths` allows reading the
same HDF5 groups from multiple file sources.

Run from repository root:
```
[uv run] python examples/zip_hdf5_dataset.py
```
"""

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# --8<-- [start:build]
import torch
from chuchichaestli.data import ZipHDF5Dataset

# select the training data in train
data_dir = "data/mnist_h5_tests/test_scenario_13/train"

zip_ds = ZipHDF5Dataset.from_groups(
    # the HDF5 files (wildcards * and ** work)
    f"{data_dir}/mnist_test_*.h5",
    # first HDF5 group
    "image/*",
    # second HDF5 group (acts as a "paired" source)
    "blurred_image/*",
    # each item is a plain (tensor_0, tensor_1) tuple
    zip_as="tuple",
    # both groups must have the same number of samples
    strict=True,
    # data type is casted when accessed
    dtype=torch.float32,
    # allocate 8M as cache for each dataset (for larger datasets,
    # the rest is read from disk)
    cache="8M",
)

print("First zip dataset")
zip_ds.datasets[0].info()
print("Second zip dataset")
zip_ds.datasets[1].info()
# --8<-- [end:build]

# Create a data loader with the dataset
batch_size = 16
shuffle = True
num_workers = 0
print(f"Creating data loader with {batch_size=}, {shuffle=}, {num_workers=}")
loader = DataLoader(
    zip_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)
# Iterate through the dataset and plot the images from each batch in one row
# and the masks from the same in another row.
for step, batch in enumerate(loader):
    images, masks = batch
    print(f"Images {step:02d}, shape={images.shape}, dtype={images.dtype}")
    print(f"Masks  {step:02d}, shape={masks.shape}, dtype={masks.dtype}")
    fig, axs = plt.subplots(ncols=len(images), nrows=2, squeeze=False)
    # plot images in first row
    for i, img in enumerate(images):
        img = img.detach()
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # plot masks in the second row
    for i, msk in enumerate(masks):
        msk = msk.detach()
        axs[1, i].imshow(msk)
        axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
print()

# After the first iteration through the dataset the items are cached
print("After first epoch...")
print("First zip dataset")
zip_ds.datasets[0].info()
print("Second zip dataset")
zip_ds.datasets[1].info()
zip_ds.close()

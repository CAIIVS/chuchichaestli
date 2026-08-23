# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Example: measuring image quality with a battery of evaluation metrics.

This script demonstrates the `EvalMetric` interface shared by all metrics in
the `metrics` module:

* `.update(data, prediction)` - register a batch and add it to the aggregate
* `.compute()`                - reduce the aggregate to a single value
* `.reset()`                  - clear the aggregate (data ranges are kept)

`MSE`, `PSNR`, and `SSIM` compare images pixel-wise, while `FID` compares
feature distributions and therefore keeps real and fake statistics apart.

Note: `FID` uses a pretrained InceptionV3 as its feature extractor, whose
weights are downloaded on first use (and cached by torchvision afterwards).

Run from repository root:
```
[uv run] python examples/metrics_evaluation.py
```
"""

# --8<-- [start:setup]
import torch
from chuchichaestli.metrics import MSE, PSNR, SSIM

torch.manual_seed(42)
batch_size, num_channels, width, height = 4, 3, 128, 128


def evaluation_batches(n_batches: int = 8):
    """Stand-in for a real evaluation loop over a model and a dataset.

    Yields (data, prediction) pairs, i.e. the target images alongside the
    imperfect reconstructions a model would produce for them.
    """
    for _ in range(n_batches):
        data = torch.rand(batch_size, num_channels, width, height)
        prediction = (data + 0.05 * torch.randn_like(data)).clamp(0, 1)
        yield data, prediction


# --8<-- [end:setup]


# --8<-- [start:battery]
metrics = [
    MSE(),
    PSNR(min_value=0, max_value=1),
    SSIM(min_value=0, max_value=1, kernel_size=7, kernel_type="gaussian"),
]

# `update` accumulates state across the entire evaluation set...
for data, prediction in evaluation_batches():
    for metric in metrics:
        metric.update(data, prediction)

# ...while `compute` reduces that aggregate once, after the loop
for metric in metrics:
    print(f"{type(metric).__name__:<4s} = {metric.compute():.4f}")
    metric.reset()
# --8<-- [end:battery]


# --8<-- [start:fid]
from chuchichaestli.metrics import FID

# FID scores feature distributions instead of pixels, so real and fake images
# are aggregated into separate statistics
fid = FID()
for data, prediction in evaluation_batches():
    fid.update(data=data, prediction=prediction)

print(f"FID  = {fid.compute():.4f}")
# --8<-- [end:fid]

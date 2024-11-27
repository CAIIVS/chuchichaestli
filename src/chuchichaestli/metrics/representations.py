import os
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch
import pathlib

def get_representation(model, batch, device, normalized=False):
    if isinstance(batch, list):
        # batch is likely list[array(images), array(labels)]
        batch = batch[0]

    if not torch.is_tensor(batch):
        # assume batch is then e.g. AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        batch = batch["pixel_values"]
        batch = batch[:, 0]

    # Convert grayscale to RGB
    if batch.ndim == 3:
        batch.unsqueeze_(1)
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)

    batch = batch.to(device)

    with torch.no_grad():
        pred = model(batch)

        if not torch.is_tensor(pred):  # Some encoders output tuples or lists
            pred = pred[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.dim() > 2:
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2)

    if normalized:
        pred = torch.nn.functional.normalize(pred, dim=-1)
    pred = pred.cpu().numpy()
    
    return pred



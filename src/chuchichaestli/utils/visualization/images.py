# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image exports, one file each, written by torchvision or matplotlib."""

from __future__ import annotations
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, get_args
import torch
from torch.utils.data import default_collate
from torchvision.utils import save_image
from chuchichaestli.data.batch import as_image_batch
from chuchichaestli.data.transforms.intensity import MinMaxScale
from chuchichaestli.utils.registry import require
from chuchichaestli.utils.tensors import as_array
from chuchichaestli.utils.visualization.base import require_mpl


__all__ = [
    "NormalizeTypes",
    "IMAGE_FORMATS",
    "save_images",
]

NormalizeTypes = Literal["image", "batch", "shared", "none"]
NORMALIZATIONS = frozenset(get_args(NormalizeTypes))
IMAGE_FORMATS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".pdf", ".svg"}
)
VECTOR_FORMATS = frozenset({".pdf", ".svg"})
UNSAFE_IN_NAME = re.compile(r"[^\w.-]+")


def save_images(
    images: Any,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    limit: int | None = 8,
    normalize: NormalizeTypes | tuple[float, float] = "image",
    cmap: str | None = None,
    draw: Callable[[Any, torch.Tensor], None] | None = None,
    title: str | None = None,
    size: float = 1.4,
    pad: float = 0.04,
    dpi: int = 150,
) -> list[Path]:
    """Write one file per image, named after the target's stem.

    Args:
        images: A tensor, a mapping of name to images, or a dataset or
            sequence of them; a dataset of tuples or dicts gives one row per
            tuple position or key, as a `Predict` stage's pairs are published.
        path: File whose stem and suffix name every image; the suffix picks
            the format.
        labels: Name per row, naming the file and the axes.
        limit: Images to take from each row, or `None` for all.
        normalize: `"image"`, `"batch"`, `"shared"`, `"none"`, or a
            `(low, high)` pair to scale into `[0, 1]` by.
        cmap: Colormap for single-channel images.
        draw: Fills the axes in place of the image, called with the axes and
            the image; `ax.get_figure()` reaches the figure for a colorbar.
        title: Heading above each image.
        size: Inches along the image's longer side; the shorter one follows
            the image's own aspect.
        pad: Inches of margin around the image.
        dpi: Output resolution.

    Returns:
        The files written.

    Raises:
        ValueError: If the suffix names no image format, if `normalize` is
            unknown, if there are not as many labels as rows, or if there is
            nothing to plot.
    """
    target = Path(path)
    require(target.suffix.lower(), IMAGE_FORMATS, "image format")
    if isinstance(normalize, str):
        require(normalize, NORMALIZATIONS, "image normalization")
    if not isinstance(images, (torch.Tensor, Mapping)):
        indices = range(len(images) if limit is None else min(len(images), limit))
        images = default_collate([images[index] for index in indices])
    if isinstance(images, Mapping):
        labels = list(images) if labels is None else labels
        images = list(images.values())
    if not isinstance(images, (list, tuple)):
        images = [images]
    batches = [as_image_batch(image)[:limit] for image in images]
    if labels is not None and len(labels) != len(batches):
        raise ValueError(
            f"An image export of {len(batches)} rows takes as many labels, "
            f"got {len(labels)}."
        )
    if not all(len(batch) for batch in batches):
        raise ValueError("An image export needs at least one image per row.")
    paths = []
    for row, batch in enumerate(_to_unit(batches, normalize)):
        label = labels[row] if labels else "" if len(batches) == 1 else str(row)
        key = UNSAFE_IN_NAME.sub("-", label)
        for index, image in enumerate(batch):
            stem = "_".join(piece for piece in (target.stem, key, str(index)) if piece)
            paths.append(target.with_name(stem + target.suffix))
            _write(
                image,
                paths[-1],
                label=label,
                cmap=cmap,
                draw=draw,
                title=title,
                size=size,
                pad=pad,
                dpi=dpi,
            )
    return paths


def _to_unit(
    batches: list[torch.Tensor], normalize: NormalizeTypes | tuple[float, float]
) -> list[torch.Tensor]:
    """Return every row mapped into `[0, 1]`.

    Args:
        batches: Images to scale, one batch per row.
        normalize: How to scale them, as `save_images` takes it.
    """
    if normalize == "shared":
        pixels = torch.cat([batch.flatten() for batch in batches])
        normalize = (pixels.min().item(), pixels.max().item())
    if normalize == "none":
        return [batch.clamp(0.0, 1.0) for batch in batches]
    if normalize == "image":
        return [torch.stack(_to_unit(list(batch), "batch")) for batch in batches]
    low, high = normalize if isinstance(normalize, (tuple, list)) else (None, None)
    scale = MinMaxScale(low, high)
    return [scale.transform(batch).nan_to_num(0.0).clamp(0.0, 1.0) for batch in batches]


def _write(
    image: torch.Tensor,
    path: Path,
    *,
    label: str,
    cmap: str | None,
    draw: Callable[[Any, torch.Tensor], None] | None,
    title: str | None,
    size: float,
    pad: float,
    dpi: int,
) -> None:
    """Write one image, as pixels alone or as a figure.

    Args:
        image: One `(C, H, W)` image in `[0, 1]`.
        path: File to write.
        label: Name for the axes, or `""` for none.
        cmap: Colormap for a single-channel image.
        draw: Fills the axes in place of the image.
        title: Heading above the image.
        size: Inches along the image's longer side.
        pad: Inches of margin around the image.
        dpi: Output resolution.
    """
    if not (cmap or draw or title or label or path.suffix.lower() in VECTOR_FORMATS):
        save_image(image, path, padding=0)
        return
    mpl = require_mpl("annotated images")
    height, width = image.shape[-2:]
    longest = max(height, width)
    figure = mpl.Figure(
        figsize=(size * width / longest + 2 * pad, size * height / longest + 2 * pad),
        layout="constrained",
    )
    figure.get_layout_engine().set(w_pad=pad, h_pad=pad)
    ax = figure.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.set_ylabel(label, fontsize=8)
    if draw is not None:
        draw(ax, image)
    elif image.shape[0] == 1:
        ax.imshow(
            as_array(image[0]),
            cmap=cmap or "gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
    else:
        ax.imshow(as_array(image.permute(1, 2, 0)), interpolation="nearest")
    if title:
        figure.suptitle(title)
    figure.savefig(path, dpi=dpi)

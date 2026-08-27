# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch dataset classes for image file reading."""

from pathlib import Path
import json
import torch
from torchvision.io import decode_image, ImageReadMode
from collections.abc import Sequence
from typing import Literal
from chuchichaestli.data.cache import nbytes
from chuchichaestli.data.base import CachingDataset, DataReturnTypes
from chuchichaestli.data.zip import ZipDataset


__all__ = ["ImageDataset", "ZipImageDataset"]


ImageReadModeKey = Literal[
    "RGB", "L", "GRAY", "GRAYSCALE", "RGBA", "RGB_ALPHA", "UNCHANGED"
]

_MODE_MAP: dict[ImageReadModeKey, ImageReadMode] = {
    "RGB": ImageReadMode.RGB,
    "L": ImageReadMode.GRAY,
    "GRAY": ImageReadMode.GRAY,
    "GRAYSCALE": ImageReadMode.GRAY,
    "RGBA": ImageReadMode.RGB_ALPHA,
    "RGB_ALPHA": ImageReadMode.RGB_ALPHA,
    "UNCHANGED": ImageReadMode.UNCHANGED,
}


class ImageTensorView:
    """Lazy, length-1 tensor-like wrapper around an image file path.

    The image is read via `torchvision.io.decode_image` and returned as a
    `uint8` tensor in `(C, H, W)` channel-first layout.  When `normalize`
    the tensor is cast to `float32` and divided by 255.
    """

    def __init__(
        self,
        path: str | Path,
        mode: ImageReadModeKey = "UNCHANGED",
        normalize: bool = True,
        apply_exif_orientation: bool = False,
    ):
        """Constructor.

        Args:
            path: image file path.
            mode: reading mode for automatic decoding; one of `"RGB"`,
                `"L"` / `"GRAY"`, `"RGBA"` / `"RGB_ALPHA"`, or `"UNCHANGED"`.
            normalize: normalize image, cast to float32.
            apply_exif_orientation: apply EXIF orientation transformation to the
                output tensor. Only applies to JPEG and PNG images.
        """
        self._path = Path(path)
        self._mode = self._parse_mode(mode) if isinstance(mode, str) else mode
        self._normalize = normalize
        self._apply_exif_orientation = apply_exif_orientation
        self._shape: tuple[int, ...] | None = None

    @staticmethod
    def _parse_mode(mode: ImageReadModeKey) -> ImageReadMode:
        """Convert a Pillow-style mode string to a `torchvision.io.ImageReadMode`.

        Args:
            mode: One of `"RGB"`, `"L"` / `"GRAY"`, `"RGBA"` / `"RGB_ALPHA"`,
                or `"UNCHANGED"`.
        """
        key = mode.upper().replace("-", "_")
        if key not in _MODE_MAP:
            raise ValueError(
                f"Unsupported image mode {mode!r}. Choose one of: {list(_MODE_MAP)}"
            )
        return _MODE_MAP[key]

    @property
    def shape(self) -> tuple[int, ...]:
        """Lazy-loading shape of the image view as `(1, C, H, W)`."""
        if self._shape is None:
            t = decode_image(
                self._path,
                mode=self._mode,
                apply_exif_orientation=self._apply_exif_orientation,
            )
            self._shape = (1, *t.shape)
        return self._shape

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Decode the image and return a tensor in `(C, H, W)` order.

        Args:
            idx: Must be `0` (single-sample view).
        """
        if idx != 0 and not isinstance(idx, slice):
            raise IndexError(f"ImageTensorView index {idx} out of range (len=1).")
        t = decode_image(
            self._path,
            mode=self._mode,
            apply_exif_orientation=self._apply_exif_orientation,
        )  # uint8, (C, H, W)
        if self._normalize:
            return t.to(torch.float32) / 255.0
        return t.to(torch.float32)


class JsonMetaView:
    """Lazy, length-1 dict view over a JSON metadata file."""

    def __init__(self, path: Path, keys: str | Sequence[str] = "*", mode: str = "r"):
        self._path = path
        self._mode = mode
        self._keys = (keys,) if isinstance(keys, str) else tuple(keys)

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        """Open and extract metadata dict (empty if it does not exist)."""
        if idx != 0:
            raise IndexError(f"JsonMetaView index {idx} out of range (len=1).")
        if not self._path.exists():
            return {}
        with self._path.open(mode=self._mode) as fh:
            if "*" in self._keys:
                return json.load(fh)
            else:
                dct = json.load(fh)
                return {k: dct[k] for k in self._keys}


class ImageDataset(CachingDataset):
    """Dataset for loading image files (JPEG, PNG, and others) with caching.

    Features:
    - lazy per-sample image decoding via torchvision.
    - normalised to `[0, 1]` by default
        (disable with `normalize=False` to keep the raw uint8 range `[0, 255]`).
    - configurable color decoding mode.
    - optional per-sample JSON metadata loading.
    - shared memory caching via CachingDataset.
    - multi-file globbing support with automatic index fusing.
    """

    FILE_EXTENSIONS: list[str] = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
    ]

    def __init__(
        self,
        path: str | Path | Sequence[str] | Sequence[Path],
        mode: str = "UNCHANGED",
        normalize: bool = True,
        attrs_keys: str | Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        cache: int | float | str | bool | nbytes | None = "4G",
        attrs_cache: int | float | str | bool | nbytes | None = None,
        preload: bool = False,
        **kwargs,
    ):
        """Constructor.

        Args:
            path: Path to image file(s), directory, or glob pattern.
                Wildcards ``*`` and ``**`` are supported.
            mode: Color decoding mode; one of: `"RGB"`, `"L"` / `"GRAY"`,
                `"RGBA"` / `"RGB_ALPHA"`, or `"UNCHANGED"`.
            normalize: If `True`, pixels have a range `[0, 1]`. If `False`,
                they keep the original range (type conversion happens after).
            attrs_keys: For each image `foo.jpg`, a corresponding `foo.json` in
                the same directory is loaded and the specified metadata keys are
                returned (use `"*"` for the entire dictionary).
                Set to `None` to disable metadata loading.
            dtype: Data tensor type; default: `torch.float32`.
            return_as: Return type; one of `['tuple', 'dict']` or a custom dict mapping.
                - `'tuple'`: Returns tuple of samples (default).
                - `'dict'`: Returns dict with key `'data'` (and
                  `'attrs'` when attributes are present).
                - `dict`: Custom mapping, e.g. `{'input': 0, 'target': 1}`.
            cache: Cache size for data items (e.g. `"4G"`, `4.0`, or bytes).
            attrs_cache: Cache size for attributes/metadata.
            preload: Preload and cache the dataset.
            kwargs: Reserved for forward-compatibility.

        Note: `sample_axis=None` is redundant; ImageDataset already treats each
            image file as exactly one sample by definition.
        """
        self.read_mode: ImageReadMode = ImageTensorView._parse_mode(mode)
        self.normalize = normalize
        self.attrs_patterns: tuple[str, ...] | None = None
        self.attrs_suffix: str = kwargs.pop("attrs_suffix", ".json")
        if attrs_keys is not None:
            self.attrs_patterns = (
                (attrs_keys,) if isinstance(attrs_keys, str) else tuple(attrs_keys)
            )

        super().__init__(
            path=path,
            dtype=dtype,
            return_as=return_as,
            cache=cache,
            attrs_cache=attrs_cache,
            preload=preload,
            sample_axis=0,
            copy_on_write=True,  # images are always fresh arrays, no mmap aliasing
            has_attrs=attrs_keys is not None,
        )

    def load(self, **kwargs) -> None:
        """Build one `ImageTensorView` per image file and index them."""
        for file_path in self.files:
            self._mmap.append(
                ImageTensorView(file_path, self.read_mode, self.normalize)
            )
            if self.has_attrs:
                mdtafile = file_path.with_suffix(self.attrs_suffix)
                self._mmap_attrs.append(
                    JsonMetaView(mdtafile, keys=self.attrs_patterns)
                )
        self._build_index()

    def close(self, **kwargs) -> None:
        """Release all views and purge the shared-memory cache."""
        self._mmap.clear()
        self._mmap_attrs.clear()
        self._file_offsets.clear()
        super().close()

    def info(self, print_: bool = True) -> str:
        """Return (and optionally print) a human-readable dataset summary.

        Args:
            print_: Whether to print the summary to stdout.
        """
        summary = str(self) + "\n"
        summary += "-" * 50 + "\n"
        summary += f"Files:              {self.n_files}\n"
        summary += f"Samples:            {self.n_samples}\n"
        summary += f"Mode:               {self.read_mode}\n"
        summary += f"Normalize:          {self.normalize}\n"
        if self.n_files:
            summary += f"Shape:              {self.shape}\n"
            summary += f"Sample shape:       {self.sample_shape}\n"
        if self.attrs_patterns:
            summary += f"Attr patterns:      {self.attrs_patterns}\n"
            summary += f"Attrs suffix:       {self.attrs_suffix}\n"
        if print_:
            print(summary)
        return summary


class ZipImageDataset(ZipDataset):
    """Zip iterator over multiple `ImageDataset` instances.

    Enables paired loading of images from different directories (e.g.
    `inputs/` and `targets/`) at the same index.
    """

    @classmethod
    def from_paths(
        cls,
        *paths: str | Path | Sequence[str] | Sequence[Path],
        mode: str = "UNCHANGED",
        normalize: bool = True,
        attrs_keys: str | Sequence[str] | None = None,
        zip_as: DataReturnTypes | None = "tuple",
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipImageDataset":
        """Create a :class:`ZipImageDataset` from multiple directory paths.

        Args:
            *paths: One path (or glob) per dataset.
            mode: Color decoding mode applied to all datasets.
            normalize: Whether to normalise pixel values to `[0, 1]`.
            attrs_keys: For each image `foo.jpg`, a corresponding `foo.json` in
                the same directory is loaded and the specified metadata keys are
                returned (use `"*"` for the entire dictionary).
                Set to `None` to disable metadata loading.
            zip_as: Return format for the combined output.
            strict: If `True`, all datasets must have the same length.
            cache: Cache budget per dataset.
            attrs_cache: Attribute cache budget per dataset.
            preload: Whether to eagerly preload all datasets.
            dtype: Tensor dtype.
            return_as: Return format for individual datasets.
            **kwargs: Extra keyword arguments forwarded to `ImageDataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided.")
        datasets = [
            ImageDataset(
                p,
                mode=mode,
                normalize=normalize,
                attrs_keys=attrs_keys,
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            for p in paths
        ]
        return cls(*datasets, zip_as=zip_as, strict=strict)

    @classmethod
    def from_named_paths(
        cls,
        paths: dict[str, str | Path | Sequence[str] | Sequence[Path]],
        mode: str = "UNCHANGED",
        normalize: bool = True,
        attrs_keys: str | Sequence[str] | None = None,
        strict: bool = False,
        cache: int | float | str | bool | None = "2G",
        attrs_cache: int | float | str | bool | None = None,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        return_as: DataReturnTypes | None = "tuple",
        **kwargs,
    ) -> "ZipImageDataset":
        """Create a :class:`ZipImageDataset` with named paths.

        Args:
            paths: Mapping of output name → path / glob.
            mode: Color decoding mode applied to all datasets.
            normalize: Whether to normalise pixel values to `[0, 1]`.
            attrs_keys: For each image `foo.jpg`, a corresponding `foo.json` in
                the same directory is loaded and the specified metadata keys are
                returned (use `"*"` for the entire dictionary).
                Set to `None` to disable metadata loading.
            strict: If `True`, all datasets must have the same length.
            cache: Cache budget per dataset.
            attrs_cache: Attribute cache budget per dataset.
            preload: Whether to eagerly preload all datasets.
            dtype: Tensor dtype.
            return_as: Return format for individual datasets.
            **kwargs: Extra keyword arguments forwarded to `ImageDataset`.
        """
        if not paths:
            raise ValueError("At least one path must be provided.")
        datasets = [
            ImageDataset(
                p,
                mode=mode,
                normalize=normalize,
                attrs_keys=attrs_keys,
                dtype=dtype,
                return_as=return_as,
                cache=cache,
                attrs_cache=attrs_cache,
                preload=preload,
                **kwargs,
            )
            for p in paths.values()
        ]
        zip_as = {name: idx for idx, name in enumerate(paths)}
        return cls(*datasets, zip_as=zip_as, strict=strict)

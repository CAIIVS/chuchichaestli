# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the ImageDataset and ZipImageDataset classes."""

from pathlib import Path
import tempfile
import json
import torch
import torchvision.io as tvio
from torchvision.io import ImageReadMode
import pytest
from chuchichaestli.data.image import (
    ImageDataset,
    ZipImageDataset,
    ImageTensorView,
    JsonMetaView,
    _MODE_MAP,
)


def _make_rgb_image(path: Path, h: int = 32, w: int = 32) -> Path:
    """Write a random uint8 RGB JPEG to path."""
    tensor = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
    tvio.write_jpeg(tensor, str(path))
    return path


def _make_gray_image(path: Path, h: int = 32, w: int = 32) -> Path:
    """Write a random uint8 greyscale PNG to path."""
    tensor = torch.randint(0, 256, (1, h, w), dtype=torch.uint8)
    tvio.write_png(tensor, str(path))
    return path


def _make_rgba_image(path: Path, h: int = 32, w: int = 32) -> Path:
    """Write a random uint8 RGBA PNG to path."""
    tensor = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
    tvio.write_png(tensor, str(path))
    return path


def _make_solid_rgb_image(path: Path, value: int, h: int = 32, w: int = 32) -> Path:
    """Write a solid-colour uint8 RGB PNG to path."""
    tensor = torch.full((3, h, w), value, dtype=torch.uint8)
    tvio.write_png(tensor, str(path))
    return path


def _make_metadata(image_path: Path, data: dict) -> Path:
    """Write a JSON metadata next to image_path."""
    mdta = image_path.with_suffix(".json")
    mdta.write_text(json.dumps(data))
    return mdta


@pytest.fixture
def temp_dir():
    """Temporary directory cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

@pytest.fixture
def single_rgb_file(temp_dir):
    """Yield the path to a single 32x32 RGB JPEG in a temporary directory."""
    return _make_rgb_image(temp_dir / "img.jpg")


@pytest.fixture
def multi_rgb_files(temp_dir):
    """Yield a list of five 32x32 RGB JPEGs in a temporary directory."""
    return [_make_rgb_image(temp_dir / f"img_{i:02d}.jpg") for i in range(5)]


@pytest.fixture
def multi_rgb_files_with_metadata(temp_dir):
    """Yield four 32x32 RGB PNGs each accompanied by a JSON metadata."""
    paths = []
    for i in range(4):
        p = _make_rgb_image(temp_dir / f"img_{i:02d}.png")
        _make_metadata(p, {"index": i, "label": f"class_{i % 2}"})
        paths.append(p)
    return paths


class TestParseMode:
    """Test ImageTensorView._parse_mode method."""

    @pytest.mark.parametrize("mode", ["RGB", "L", "GRAY", "RGBA", "UNCHANGED"])
    def test_modes(self, mode):
        """Test various modes."""
        out = ImageTensorView._parse_mode(mode)
        assert isinstance(out, ImageReadMode)
        assert out in _MODE_MAP.values()

    def test_case_insensitive(self):
        """Test case-insensitive mode."""
        assert ImageTensorView._parse_mode("rgb") == ImageReadMode.RGB

    def test_invalid_raises(self):
        """Test invalid mode."""
        with pytest.raises(ValueError, match="Unsupported"):
            ImageTensorView._parse_mode("XYZ")


class TestImageTensorView:
    """Tests for the internal ImageTensorView lazy-image wrapper."""

    def test_shape_rgb(self, single_rgb_file):
        """Shape of an RGB view is (1, 3, H, W)."""
        view = ImageTensorView(single_rgb_file, mode="RGB")
        assert view.shape == (1, 3, 32, 32)

    def test_shape_gray(self, temp_dir):
        """Shape of a greyscale view is (1, 1, H, W)."""
        p = _make_gray_image(temp_dir / "gray.png")
        view = ImageTensorView(p, mode="GRAY")
        assert view.shape == (1, 1, 32, 32)

    def test_len(self, single_rgb_file):
        """A single-image view always has length 1."""
        assert len(ImageTensorView(single_rgb_file, mode="RGB")) == 1

    def test_getitem_zero_returns_tensor(self, single_rgb_file):
        """Index 0 returns a float32 tensor with the expected spatial shape."""
        view = ImageTensorView(single_rgb_file, mode="RGB", normalize=True)
        t = view[0]
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert t.shape == (3, 32, 32)

    def test_getitem_normalized_range(self, single_rgb_file):
        """With normalize=True pixel values lie in [0, 1]."""
        view = ImageTensorView(single_rgb_file, mode="RGB", normalize=True)
        t = view[0]
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_getitem_unnormalized_range(self, single_rgb_file):
        """With normalize=False pixel values lie in [0, 255] as float32."""
        view = ImageTensorView(single_rgb_file, mode="RGB", normalize=False)
        t = view[0]
        assert t.dtype == torch.float32
        assert t.max() <= 255.0

    def test_getitem_out_of_range(self, single_rgb_file):
        """Any index other than 0 raises IndexError."""
        view = ImageTensorView(single_rgb_file, mode="RGB")
        with pytest.raises(IndexError):
            _ = view[1]

    def test_shape_cached_after_first_access(self, single_rgb_file):
        """The shape tuple is cached after the first property access."""
        view = ImageTensorView(single_rgb_file, mode="RGB")
        _ = view.shape
        assert view._shape is not None
        assert view.shape == (1, 3, 32, 32)

    def test_channel_first_layout(self, single_rgb_file):
        """Channels must be axis-0 (CHW), not axis-2 (HWC)."""
        view = ImageTensorView(single_rgb_file, mode="RGB")
        assert view[0].shape[0] == 3

class TestJsonMetaView:
    """Tests for the internal JsonMetaView metadata wrapper."""

    def test_len(self, temp_dir):
        """A metadata view always has length 1."""
        p = temp_dir / "meta.json"
        p.write_text('{"a": 1}')
        assert len(JsonMetaView(p)) == 1

    def test_shape(self, temp_dir):
        """The shape of a sidecar view is always (1,)."""
        p = temp_dir / "meta.json"
        p.write_text('{}')
        assert JsonMetaView(p).shape == (1,)

    def test_getitem_existing(self, temp_dir):
        """Index 0 returns the parsed JSON dict when the file exists."""
        data = {"label": "cat", "confidence": 0.99}
        p = temp_dir / "meta.json"
        p.write_text(json.dumps(data))
        assert JsonMetaView(p)[0] == data

    def test_getitem_keys(self, temp_dir):
        """Keys returns the parsed JSON dict subset."""
        data = {"a": 1, "b": 2, "c": 3}
        p = temp_dir / "meta.json"
        p.write_text(json.dumps(data))
        view = JsonMetaView(p, keys=("b", "c"))
        assert "a" not in view[0]
        assert "b" in view[0]
        assert "c" in view[0]

    def test_getitem_all_keys(self, temp_dir):
        """Keys returns the parsed JSON dict subset."""
        data = {"a": 1, "b": 2, "c": 3}
        p = temp_dir / "meta.json"
        p.write_text(json.dumps(data))
        view = JsonMetaView(p, keys="*")
        assert "a" in view[0]
        assert "b" in view[0]
        assert "c" in view[0]

    def test_getitem_missing_file(self, temp_dir):
        """Index 0 returns an empty dict when the sidecar file is absent."""
        assert JsonMetaView(temp_dir / "missing.json")[0] == {}

    def test_getitem_out_of_range(self, temp_dir):
        """Any index other than 0 raises IndexError."""
        p = temp_dir / "meta.json"
        p.write_text('{}')
        with pytest.raises(IndexError):
            JsonMetaView(p)[1]


class TestImageDataset:
    """Tests for ImageDataset class."""

    def test_single_file_len(self, single_rgb_file):
        """A dataset built from one file has length 1."""
        with ImageDataset(single_rgb_file) as ds:
            assert len(ds) == 1

    def test_multi_file_len(self, multi_rgb_files):
        """A dataset built from five files has length 5."""
        with ImageDataset(multi_rgb_files) as ds:
            assert len(ds) == 5

    def test_n_files(self, multi_rgb_files):
        """n_files equals the number of image files provided."""
        with ImageDataset(multi_rgb_files) as ds:
            assert ds.n_files == 5

    def test_glob_pattern(self, temp_dir):
        """A glob pattern discovers all matching files in the directory."""
        for i in range(3):
            _make_rgb_image(temp_dir / f"x_{i}.jpg")
        with ImageDataset(str(temp_dir / "*.jpg")) as ds:
            assert len(ds) == 3

    def test_recursive_glob(self, temp_dir):
        """A ** glob pattern discovers files in nested subdirectories."""
        sub = temp_dir / "sub"
        sub.mkdir()
        _make_rgb_image(temp_dir / "a.jpg")
        _make_rgb_image(sub / "b.jpg")
        with ImageDataset(str(temp_dir / "**" / "*.jpg")) as ds:
            assert len(ds) == 2

    def test_default_dtype(self, single_rgb_file):
        """The default output dtype is torch.float32."""
        with ImageDataset(single_rgb_file) as ds:
            assert ds.dtype == torch.float32

    def test_custom_dtype(self, single_rgb_file):
        """Specifying dtype=torch.float64 produces float64 sample tensors."""
        with ImageDataset(single_rgb_file, dtype=torch.float64) as ds:
            assert ds[0].dtype == torch.float64

    def test_file_extensions(self):
        """FILE_EXTENSIONS includes at minimum .jpg, .jpeg, and .png."""
        exts = ImageDataset.FILE_EXTENSIONS
        assert ".jpg" in exts
        assert ".jpeg" in exts
        assert ".png" in exts
        assert ".gif" in exts
        assert ".webp" in exts

    def test_close_clears_state(self, multi_rgb_files):
        """close() empties _mmap, _mmap_attrs, and _file_offsets."""
        ds = ImageDataset(multi_rgb_files)
        ds.close()
        assert ds._mmap == []
        assert ds._mmap_attrs == []
        assert ds._file_offsets == []

    def test_context_manager(self, single_rgb_file):
        """The context manager calls close() on exit, clearing _mmap."""
        with ImageDataset(single_rgb_file) as ds:
            _ = ds[0]
        assert ds._mmap == []

    def test_returns_tensor(self, single_rgb_file):
        """Each sample is returned as a torch.Tensor."""
        with ImageDataset(single_rgb_file) as ds:
            assert isinstance(ds[0], torch.Tensor)

    def test_shape_rgb(self, single_rgb_file):
        """An RGB sample has shape (3, H, W)."""
        with ImageDataset(single_rgb_file) as ds:
            assert ds[0].shape == (3, 32, 32)

    def test_shape_grayscale(self, temp_dir):
        """A greyscale sample loaded with mode='L' has shape (1, H, W)."""
        p = _make_gray_image(temp_dir / "gray.png")
        with ImageDataset(p, mode="L") as ds:
            assert ds[0].shape == (1, 32, 32)

    def test_normalized_range(self, single_rgb_file):
        """With normalize=True all pixel values lie in [0, 1]."""
        with ImageDataset(single_rgb_file, normalize=True) as ds:
            t = ds[0]
            assert t.min().item() >= 0.0
            assert t.max().item() <= 1.0 + 1e-6

    def test_unnormalized_range(self, single_rgb_file):
        """With normalize=False all pixel values lie in [0, 255]."""
        with ImageDataset(single_rgb_file, normalize=False) as ds:
            assert ds[0].max().item() <= 255.0 + 1e-6

    def test_multifile_indexing(self, multi_rgb_files):
        """Every index across a multi-file dataset returns a correctly shaped tensor."""
        with ImageDataset(multi_rgb_files) as ds:
            for i in range(len(ds)):
                assert ds[i].shape == (3, 32, 32)

    def test_negative_index(self, multi_rgb_files):
        """Negative indices wrap around to the end of the dataset."""
        with ImageDataset(multi_rgb_files) as ds:
            assert ds[-1].shape == (3, 32, 32)

    def test_index_out_of_range(self, single_rgb_file):
        """A positive index beyond the dataset length raises IndexError."""
        with ImageDataset(single_rgb_file) as ds:
            with pytest.raises(IndexError):
                _ = ds[10]

    def test_large_negative_index_out_of_range(self, single_rgb_file):
        """A large negative index beyond the dataset length raises IndexError."""
        with ImageDataset(single_rgb_file) as ds:
            with pytest.raises(IndexError):
                _ = ds[-10]

    def test_pixel_values_round_trip(self, temp_dir):
        """A known pixel value must survive the encode → load cycle (lossless PNG)."""
        p = _make_solid_rgb_image(temp_dir / "solid.png", value=200)
        with ImageDataset(p, normalize=False) as ds:
            assert torch.allclose(ds[0], torch.full((3, 32, 32), 200.0))

    def test_rgb_channels(self, single_rgb_file):
        """mode='RGB' produces a 3-channel output tensor."""
        with ImageDataset(single_rgb_file, mode="RGB") as ds:
            assert ds[0].shape[0] == 3

    def test_grayscale_channels(self, single_rgb_file):
        """Any RGB image converted to mode='L' yields a 1-channel tensor."""
        with ImageDataset(single_rgb_file, mode="L") as ds:
            assert ds[0].shape[0] == 1

    def test_rgba_channels(self, temp_dir):
        """mode='RGBA' on a PNG with an alpha channel yields a 4-channel tensor."""
        p = _make_rgba_image(temp_dir / "rgba.png")
        with ImageDataset(p, mode="RGBA") as ds:
            assert ds[0].shape[0] == 4

    def test_invalid_mode_raises(self, single_rgb_file):
        """An unrecognised mode string raises ValueError at construction time."""
        with pytest.raises(ValueError, match="Unsupported"):
            ImageDataset(single_rgb_file, mode="XYZ")

    def test_metadata_loaded(self, multi_rgb_files_with_metadata):
        """When metadata exist the attrs dict is populated and returned alongside the image."""
        with ImageDataset(
            multi_rgb_files_with_metadata,
            attrs_keys="*",
            return_as="tuple",
        ) as ds:
            out = ds[0]
            item, attrs = out
            assert isinstance(item, torch.Tensor)
            assert isinstance(attrs, dict)
            assert "index" in attrs

    def test_missing_metadata_returns_empty_dict(self, multi_rgb_files):
        """A missing metadata file silently produces an empty attrs dict."""
        with ImageDataset(
            multi_rgb_files,
            attrs_keys="*",
            return_as="tuple",
        ) as ds:
            _, attrs = ds[0]
            assert attrs == {}

    def test_attrs_disabled(self, multi_rgb_files_with_metadata):
        """attrs_keys=None disables metadata loading and returns a plain tensor."""
        with ImageDataset(multi_rgb_files_with_metadata, attrs_keys=None) as ds:
            assert isinstance(ds[0], torch.Tensor)

    def test_attrs_dict_return(self, multi_rgb_files_with_metadata):
        """return_as='dict' wraps both image and attrs in a single dictionary."""
        with ImageDataset(
            multi_rgb_files_with_metadata,
            attrs_keys="*",
            return_as="dict",
        ) as ds:
            item = ds[0]
            assert isinstance(item, dict)
            assert "data" in item
            assert "attrs" in item

    def test_custom_attrs_suffix(self, temp_dir):
        """A custom attrs_suffix reads metadata from a non-.json extension."""
        p = _make_rgb_image(temp_dir / "img.png")
        (temp_dir / "img.meta").write_text('{"custom": true}')
        with ImageDataset(p, attrs_keys="*", attrs_suffix=".meta", return_as="tuple") as ds:
            _, attrs = ds[0]
            assert attrs.get("custom") is True

    def test_preload(self, multi_rgb_files):
        """preload=True fills the cache at construction so n_cached > 0."""
        with ImageDataset(multi_rgb_files, cache="10M", preload=True) as ds:
            assert ds.n_cached > 0

    def test_cache_none_disables_cache(self, single_rgb_file):
        """cache=None disables caching without breaking item access."""
        with ImageDataset(single_rgb_file, cache=None) as ds:
            assert isinstance(ds[0], torch.Tensor)
            assert ds.n_cached == 0

    def test_repeated_access_consistent(self, single_rgb_file):
        """Accessing the same index twice returns identical tensors."""
        with ImageDataset(single_rgb_file) as ds:
            assert torch.allclose(ds[0].clone(), ds[0].clone())

    def test_dataloader_batch_shape(self, multi_rgb_files):
        """The first batch from a DataLoader has the expected (B, C, H, W) shape."""
        with ImageDataset(multi_rgb_files) as ds:
            loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            assert isinstance(batch, torch.Tensor)
            assert batch.shape == (4, 3, 32, 32)
            assert ds.n_cached > 0

    def test_info_returns_string(self, single_rgb_file):
        """info() returns a non-empty string containing key field names."""
        with ImageDataset(single_rgb_file) as ds:
            summary = ds.info(print_=False)
            assert isinstance(summary, str)
            assert "Files" in summary
            assert "Samples" in summary

    def test_info_prints(self, single_rgb_file, capsys):
        """info(print_=True) writes output to stdout."""
        with ImageDataset(single_rgb_file) as ds:
            ds.info(print_=True)
        assert capsys.readouterr().out


class TestZipImageDataset:
    """Tests for ZipImageDataset and its factory class methods."""

    def test_from_paths_basic(self, temp_dir):
        """from_paths() zips two directories into paired tuple samples."""
        dir_a = temp_dir / "a"
        dir_b = temp_dir / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        for i in range(3):
            _make_rgb_image(dir_a / f"{i}.jpg")
            _make_rgb_image(dir_b / f"{i}.jpg")

        with ZipImageDataset.from_paths(
            str(dir_a / "*.jpg"), str(dir_b / "*.jpg"), zip_as="tuple"
        ) as ds:
            assert len(ds) == 3
            item = ds[0]
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_from_named_paths(self, temp_dir):
        """from_named_paths() returns dict samples whose keys match the supplied names."""
        dir_a = temp_dir / "inputs"
        dir_b = temp_dir / "targets"
        dir_a.mkdir()
        dir_b.mkdir()
        for i in range(4):
            _make_rgb_image(dir_a / f"{i}.png")
            _make_rgb_image(dir_b / f"{i}.png")

        with ZipImageDataset.from_named_paths(
            {"input": str(dir_a / "*.png"), "target": str(dir_b / "*.png")},
            strict=True,
        ) as ds:
            assert len(ds) == 4
            item = ds[0]
            assert isinstance(item, dict)
            assert set(item.keys()) == {"input", "target"}

    def test_from_paths_strict_length_mismatch(self, temp_dir):
        """from_paths() with strict=True raises ValueError when dataset lengths differ."""
        dir_a = temp_dir / "a"
        dir_a.mkdir()
        dir_b = temp_dir / "b"
        dir_b.mkdir()
        for i in range(3):
            _make_rgb_image(dir_a / f"{i}.jpg")
        for i in range(5):
            _make_rgb_image(dir_b / f"{i}.jpg")

        with pytest.raises(ValueError, match="same length"):
            ZipImageDataset.from_paths(
                str(dir_a / "*.jpg"), str(dir_b / "*.jpg"), strict=True
            )

    def test_from_paths_no_paths_raises(self):
        """from_paths() with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipImageDataset.from_paths()

    def test_from_named_paths_empty_raises(self):
        """from_named_paths() with an empty dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipImageDataset.from_named_paths({})

    def test_close_propagates(self, temp_dir):
        """close() on the zip dataset propagates to each constituent ImageDataset."""
        dir_a = temp_dir / "a"
        dir_a.mkdir()
        _make_rgb_image(dir_a / "img.jpg")
        ds = ZipImageDataset.from_paths(str(dir_a / "*.jpg"))
        ds.close()
        for sub in ds.datasets:
            assert sub._mmap == []


# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the SafetensorsDataset and ZipSafetensorsDataset classes."""

from pathlib import Path
import tempfile
import torch
import warnings
import pytest
from safetensors import safe_open
from safetensors.torch import save_file as st_save
from chuchichaestli.data.safetensors import (
    SafetensorsDataset,
    ZipSafetensorsDataset,
    SafetensorsView,
)


def _save(path: Path, **tensors: torch.Tensor) -> Path:
    """Save *arrays* as a .safetensors file at *path*."""
    st_save({k: v for k, v in tensors.items()}, str(path))
    return path


@pytest.fixture
def view_file(tmp_path):
    """A .safetensors file purpose-built for _SafetensorsView unit tests.

    Contains three keys with known, reproducible values:
      - 'a': shape (4, 3)  – values 0..11
      - 'b': shape (6, 3)  – values 100..117
      - 'c': shape (2, 3)  – values 200..205
    All float32 so dtype comparisons are unambiguous.
    """
    path = tmp_path / "view_test.safetensors"
    _save(
        path,
        a=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        b=(torch.arange(18, dtype=torch.float32) + 100).reshape(6, 3),
        c=(torch.arange(6, dtype=torch.float32) + 200).reshape(2, 3),
    )
    return path


@pytest.fixture
def view_handle(view_file):
    """Open safe_open handle for view_file (closed at test teardown)."""
    handle = safe_open(str(view_file), framework="pt")
    yield handle


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def single_file(temp_dir):
    """Single .safetensors file with several tensors."""
    return _save(
        temp_dir / "dataset.safetensors",
        images=torch.randn(100, 3, 64, 64),
        images_part2=torch.randn(100, 3, 64, 64),
        features=torch.randn(100, 128),
        labels=torch.randint(0, 10, (100,), dtype=torch.int32),
        bbox=torch.rand(100, 4),
    )


@pytest.fixture
def multiple_files(temp_dir):
    """Three .safetensors files, each with 50 samples."""
    files = []
    for i in range(3):
        files.append(
            _save(
                temp_dir / f"data_{i}.safetensors",
                data=torch.randn(50, 3, 32, 32),
            )
        )
    return files


class TestSafetensorsView:
    """Test the SafetensorsView class."""

    @staticmethod
    def _make_view(handle, keys: list[str]) -> SafetensorsView:
        """Build a _SafetensorsView from `handle` for the given `keys`."""
        key_lengths = [handle.get_slice(k)[:].shape[0] for k in keys]
        sample_shape = handle.get_slice(keys[0])[:].shape[1:]
        dtype = handle.get_slice(keys[0])[:].dtype
        return SafetensorsView(handle, keys, key_lengths, sample_shape, dtype)

    def test_shape_single_key(self, view_handle):
        """Shape reports (N,) + sample_shape for a single key."""
        view = self._make_view(view_handle, ["a"])
        assert view.shape == (4, 3)

    def test_shape_multi_key(self, view_handle):
        """Shape sums N across all keys while keeping sample_shape."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        assert view.shape == (12, 3)  # 4 + 6 + 2

    def test_dtype(self, view_handle):
        """Data type matches the torch.dtype of the underlying tensors."""
        view = self._make_view(view_handle, ["a"])
        assert view.dtype == torch.float32

    def test_len_single_key(self, view_handle):
        """Length returns the number of samples for a single key."""
        view = self._make_view(view_handle, ["b"])
        assert len(view) == 6

    def test_len_multi_key(self, view_handle):
        """Length returns the total sample count across all keys."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        assert len(view) == 12

    def test_offsets_computed_correctly(self, view_handle):
        """Internal _offsets list reflects cumulative key boundaries."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        assert view._offsets == [0, 4, 10]

    def test_getitem_first_sample(self, view_handle):
        """Index 0 returns the first row of key 'a'."""
        view = self._make_view(view_handle, ["a"])
        result = view[0]
        expected = torch.tensor([0.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_last_sample_single_key(self, view_handle):
        """Index N-1 returns the last row of the only key."""
        view = self._make_view(view_handle, ["a"])
        result = view[3]  # 4th of 4 rows in 'a'
        expected = torch.tensor([9.0, 10.0, 11.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_returns_tensor(self, view_handle):
        """__getitem__ always returns a torch.Tensor."""
        view = self._make_view(view_handle, ["a"])
        assert isinstance(view[0], torch.Tensor)

    def test_getitem_sample_shape(self, view_handle):
        """Each returned sample has the correct shape (no batch dimension)."""
        view = self._make_view(view_handle, ["a"])
        assert view[0].shape == (3,)

    def test_getitem_dispatches_to_first_key(self, view_handle):
        """Indices in [0, 4) resolve to key 'a'."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        result = view[2]
        expected = torch.tensor([6.0, 7.0, 8.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_dispatches_to_second_key(self, view_handle):
        """Indices in [4, 10) resolve to key 'b'."""
        view = self._make_view(view_handle, ["a", "b"])
        result = view[4]
        expected = torch.tensor([100.0, 101.0, 102.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_dispatches_to_third_key(self, view_handle):
        """Indices in [10, 12) resolve to key 'c'."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        result = view[10]
        expected = torch.tensor([200.0, 201.0, 202.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_boundary_between_keys(self, view_handle):
        """The index immediately before a key boundary stays in the prior key."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        result = view[3]
        expected = torch.tensor([9.0, 10.0, 11.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_last_sample_multi_key(self, view_handle):
        """The last global index resolves correctly to the last key."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        result = view[11]
        expected = torch.tensor([203.0, 204.0, 205.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_all_samples_values(self, view_handle):
        """Iterating every index returns values matching the original arrays."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        expected_concat = torch.cat(
            [
                torch.arange(12, dtype=torch.float32).reshape(4, 3),
                (torch.arange(18, dtype=torch.float32) + 100).reshape(6, 3),
                (torch.arange(6, dtype=torch.float32) + 200).reshape(2, 3),
            ],
            dim=0,
        )
        for i in range(len(view)):
            torch.testing.assert_close(view[i], expected_concat[i])

    def test_getitem_negative_minus_one(self, view_handle):
        """Index -1 is equivalent to the last sample."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        torch.testing.assert_close(view[-1], view[11])

    def test_getitem_negative_minus_total(self, view_handle):
        """Index -N is equivalent to index 0."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        torch.testing.assert_close(view[-12], view[0])

    def test_getitem_negative_mid_range(self, view_handle):
        """A mid-range negative index resolves to the expected positive index."""
        view = self._make_view(view_handle, ["a", "b", "c"])
        # -3 → index 9, which is in key 'b' local index 5 → row [115, 116, 117]
        result = view[-3]
        expected = torch.tensor([115.0, 116.0, 117.0])
        torch.testing.assert_close(result, expected)

    def test_getitem_out_of_range_positive(self, view_handle):
        """A positive index ≥ len raises IndexError."""
        view = self._make_view(view_handle, ["a"])
        with pytest.raises(IndexError):
            _ = view[4]

    def test_getitem_out_of_range_negative(self, view_handle):
        """A negative index < -len raises IndexError."""
        view = self._make_view(view_handle, ["a"])
        with pytest.raises(IndexError):
            _ = view[-5]

    def test_getitem_empty_dataset(self, view_handle):
        """A view with total length 0 raises IndexError for any index."""
        view = SafetensorsView(
            view_handle,
            keys=[],
            key_lengths=[],
            sample_shape=(3,),
            dtype=torch.float32,
        )
        assert len(view) == 0
        with pytest.raises(IndexError):
            _ = view[0]

    def test_single_sample_key(self, tmp_path, view_handle):
        """A key containing exactly one sample is indexed correctly."""
        path = tmp_path / "one_sample.safetensors"
        _save(path, only=torch.tensor([[1.0, 2.0, 3.0]]))
        handle = safe_open(str(path), framework="pt")
        view = self._make_view(handle, ["only"])
        assert len(view) == 1
        torch.testing.assert_close(view[0], torch.tensor([1.0, 2.0, 3.0]))
        with pytest.raises(IndexError):
            _ = view[1]

class TestSafetensorsDataset:
    """Test the SafetensorsDataset class."""

    def test_init_single_file_single_key(self, single_file):
        """Initialisation with one key selects the right tensor."""
        ds = SafetensorsDataset(single_file, keys="images")
        assert ds.n_files == 1
        assert ds.n_datasets == 1
        assert len(ds) == 100
        assert ds.shape == (100, 3, 64, 64)
        ds.close()

    def test_init_wildcard_all_keys_same_shape(self, temp_dir):
        """Wildcard selects all keys; incompatible shapes raise ValueError."""
        path = _save(
            temp_dir / "same_shape.safetensors",
            a=torch.randn(40, 3, 32, 32),
            b=torch.randn(60, 3, 32, 32),
        )
        ds = SafetensorsDataset(path, keys="*")
        assert len(ds) == 100  # 40 + 60 concatenated
        ds.close()

    def test_init_wildcard_incompatible_shapes_raises(self, single_file):
        """Selecting keys with mismatched sample shapes raises ValueError."""
        with pytest.raises(ValueError, match="incompatible"):
            SafetensorsDataset(single_file, keys="*")

    def test_getitem_returns_tensor(self, single_file):
        """__getitem__ returns a torch.Tensor of the correct shape."""
        ds = SafetensorsDataset(single_file, keys="images")
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 64, 64)
        ds.close()

    def test_getitem_values_match_source(self, temp_dir):
        """Values returned by __getitem__ match the saved array."""
        data = torch.arange(50, dtype=torch.float32).reshape(10, 5)
        path = _save(temp_dir / "known.safetensors", arr=data)
        ds = SafetensorsDataset(path, keys="arr", dtype=torch.float32)
        for i in range(10):
            assert torch.allclose(ds[i], data[i])
        ds.close()

    def test_multiple_files_fuse_index(self, multiple_files):
        """Loading multiple files fuses their indices correctly."""
        ds = SafetensorsDataset(multiple_files, keys="data")
        assert ds.n_files == 3
        assert len(ds) == 150  # 3 × 50
        for idx in [0, 50, 100]:
            assert ds[idx].shape == (3, 32, 32)
        ds.close()

    def test_glob_pattern_resolves_files(self, multiple_files):
        """Wildcard path pattern resolves to matching files."""
        directory = multiple_files[0].parent
        ds = SafetensorsDataset(str(directory / "data_*.safetensors"), keys="data")
        assert ds.n_files == 3
        assert len(ds) == 150
        ds.close()

    def test_attrs_keys_returned_with_data(self, single_file):
        """attrs_keys causes __getitem__ to return (data, attrs) tuple."""
        ds = SafetensorsDataset(
            single_file,
            keys="images",
            attrs_keys="labels",
            return_as="tuple",
        )
        assert ds.has_attrs
        ret = ds[0]
        assert isinstance(ret, tuple)
        data, attrs = ret
        assert data.shape == (3, 64, 64)
        assert isinstance(attrs, torch.Tensor)
        assert attrs.shape == torch.Size([])
        ds.close()

    def test_attrs_keys_excluded_from_data(self, single_file):
        """Keys claimed by attrs_keys must not appear in data selection."""
        path = single_file
        # Use a file where all sample shapes match so wildcard data works
        p2 = _save(
            Path(path).parent / "split.safetensors",
            data=torch.randn(20, 4, 4),
            meta=torch.arange(20, dtype=torch.int32).reshape(20, 1),
        )
        ds = SafetensorsDataset(p2, keys="data", attrs_keys="meta")
        assert len(ds) == 20
        ret = ds[0]
        assert isinstance(ret, tuple)
        data, attrs = ret
        assert data.shape == (4, 4)
        assert isinstance(attrs, torch.Tensor)
        assert attrs.shape == torch.Size([1])  # meta shape is (20, 1), so per-sample is (1,)
        ds.close()

    def test_caching(self, single_file):
        """Caching stores samples after first access."""
        ds = SafetensorsDataset(single_file, keys="images", cache="100M")
        assert ds.n_cached == 0
        _ = ds[0]
        assert ds.n_cached == 1
        _ = ds[0]
        assert ds.n_cached == 1  # no double-caching
        ds.close()

    def test_preload(self, temp_dir):
        """Preloading fills the cache up to its capacity."""
        path = _save(
            temp_dir / "small.safetensors",
            data=torch.randn(10, 8, 8),
        )
        ds = SafetensorsDataset(path, keys="data", cache="10M", preload=True)
        assert ds.n_cached == min(10, ds.n_cacheable)
        ds.close()

    def test_context_manager(self, single_file):
        """Context manager closes the dataset on exit."""
        with SafetensorsDataset(single_file, keys="images") as ds:
            assert len(ds) == 100
        # After exit the mmap list should be empty
        assert ds._mmap == []

    def test_close_clears_state(self, single_file):
        """close() empties all internal state lists."""
        ds = SafetensorsDataset(single_file, keys="images")
        ds.close()
        assert ds._mmap == []
        assert ds._mmap_attrs == []
        assert ds._file_offsets == []
        assert ds.st_buffers == []

    def test_dataloader_integration(self, single_file):
        """SafetensorsDataset works inside a PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        ds = SafetensorsDataset(single_file, keys="images")
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (8, 3, 64, 64)
        ds.close()

    def test_info(self, single_file, capsys):
        """info() prints and returns a summary string."""
        ds = SafetensorsDataset(single_file, keys="images")
        summary = ds.info(print_=True)
        captured = capsys.readouterr()
        assert "Files" in summary
        assert "Samples" in summary
        assert captured.out  # something was printed
        ds.close()

    def test_no_matching_keys_loads_nothing(self, single_file):
        """A pattern matching no keys leaves the dataset empty."""
        ds = SafetensorsDataset(single_file, keys="nonexistent_*")
        assert len(ds) == 0
        ds.close()

    def test_dtype_cast_via_base_class(self, temp_dir):
        """Data type conversion is applied by the base class via .to(dtype)."""
        data = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        path = _save(temp_dir / "cast.safetensors", arr=data)
        ds = SafetensorsDataset(path, keys="arr", dtype=torch.float64)
        sample = ds[0]
        assert sample.dtype == torch.float64
        assert sample.shape == (5,)
        ds.close()


class TestZipSafetensorsDataset:
    """Test the ZipSafetensorsDataset class."""

    def test_from_keys_same_file(self, single_file):
        """from_keys reads multiple keys from the same file in parallel."""
        ds = ZipSafetensorsDataset.from_keys(single_file, "images", "labels")
        assert len(ds) == 100
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (3, 64, 64)
        ds.close()

    def test_from_keys_dict_return(self, single_file):
        """from_keys with zip_as dict produces dict output."""
        ds = ZipSafetensorsDataset.from_keys(
            single_file,
            "images",
            "labels",
            zip_as={"image": 0, "label": 1},
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "image" in sample and "label" in sample
        assert sample["image"].shape == (3, 64, 64)
        ds.close()

    def test_from_paths_different_files(self, multiple_files):
        """from_paths reads from different files in parallel."""
        ds = ZipSafetensorsDataset.from_paths(*multiple_files, keys="data")
        assert len(ds) == 50  # limited by shortest / all same here
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 3
        ds.close()

    def test_from_named_keys(self, single_file):
        """from_named_keys returns a dict with user-supplied key names."""
        ds = ZipSafetensorsDataset.from_named_keys(
            single_file, keys={"image": "images", "label": "labels"}
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "label"}
        ds.close()

    def test_from_named_paths(self, multiple_files):
        """from_named_paths maps names to file paths and returns a dict."""
        ds = ZipSafetensorsDataset.from_named_paths(
            paths={"file0": multiple_files[0], "file1": multiple_files[1]},
            keys="data",
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"file0", "file1"}
        ds.close()

    def test_strict_length_checking(self, temp_dir):
        """Strict mode raises ValueError when datasets have different lengths."""
        short = _save(temp_dir / "short.safetensors", data=torch.randn(50, 4))
        long_ = _save(temp_dir / "long.safetensors", data=torch.randn(100, 4))
        with pytest.raises(ValueError, match="same length"):
            ZipSafetensorsDataset.from_paths(short, long_, keys="data", strict=True)
        # Non-strict: length limited by the shortest
        ds = ZipSafetensorsDataset.from_paths(short, long_, keys="data", strict=False)
        assert len(ds) == 50
        ds.close()

    def test_from_keys_empty_raises(self, single_file):
        """from_keys with no keys raises ValueError."""
        with pytest.raises(ValueError, match="At least one dataset"):
            ZipSafetensorsDataset.from_keys(single_file)

    def test_from_paths_empty_raises(self):
        """from_paths with no paths raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipSafetensorsDataset.from_paths()

    def test_from_named_keys_empty_raises(self, single_file):
        """from_named_keys with empty dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one key"):
            ZipSafetensorsDataset.from_named_keys(single_file, keys={})

    def test_from_named_paths_empty_raises(self):
        """from_named_paths with empty dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipSafetensorsDataset.from_named_paths(paths={})

    def test_caching_in_zip(self, single_file):
        """Each constituent SafetensorsDataset in a Zip populates its own cache."""
        ds = ZipSafetensorsDataset.from_keys(
            single_file,
            "images",
            "labels",
            zip_as={"image": 0, "label": 1},
            cache="100M",
        )
        sample1 = ds[0]
        for sub_ds in ds.datasets:
            if hasattr(sub_ds, "n_cached"):
                assert sub_ds.n_cached > 0
        assert isinstance(sample1, dict)
        ds.close()

    def test_dataloader_integration(self, single_file):
        """ZipSafetensorsDataset works inside a PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        ds = ZipSafetensorsDataset.from_keys(single_file, "images", "features")
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert isinstance(batch, (tuple, list))
        assert batch[0].shape == (8, 3, 64, 64)
        assert batch[1].shape == (8, 128)
        ds.close()

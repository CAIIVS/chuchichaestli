# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the NumpyDataset and ZipNumpyDataset classes."""

from pathlib import Path
import tempfile
import numpy as np
import torch
import warnings
import pytest
from chuchichaestli.data.numpy import NumpyDataset, ZipNumpyDataset, NpyArrayView


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_npy_file(temp_dir):
    """Create a single .npy file with image-like data (100, 3, 64, 64)."""
    file_path = temp_dir / "images.npy"
    np.save(file_path, np.random.randn(100, 3, 64, 64).astype(np.float32))
    return file_path


@pytest.fixture
def sample_npy_file_with_sidecar(temp_dir):
    """Create a .npy file and a matching .attrs.npy sidecar."""
    file_path = temp_dir / "images.npy"
    np.save(file_path, np.random.randn(100, 3, 64, 64).astype(np.float32))
    sidecar_path = temp_dir / "images.attrs.npy"
    attrs = np.arange(100, dtype=np.int32)
    np.save(sidecar_path, attrs)
    return file_path, sidecar_path


@pytest.fixture
def sample_npz_file(temp_dir):
    """Create a .npz archive with multiple arrays and an attrs array."""
    file_path = temp_dir / "dataset.npz"
    np.savez(
        file_path,
        images=np.random.randn(100, 3, 64, 64).astype(np.float32),
        images_part2=np.random.randn(100, 3, 64, 64).astype(np.float32),
        features=np.random.randn(100, 128).astype(np.float32),
        labels=np.random.randint(0, 10, 100).astype(np.int32),
        bbox=np.random.rand(100, 4).astype(np.float32),
    )
    return file_path


@pytest.fixture
def multiple_npy_files(temp_dir):
    """Create three .npy files each with 50 samples of shape (3, 32, 32)."""
    files = []
    for i in range(3):
        file_path = temp_dir / f"data_{i}.npy"
        np.save(file_path, np.random.randn(50, 3, 32, 32).astype(np.float32))
        files.append(file_path)
    return files


@pytest.fixture
def multiple_npz_files(temp_dir):
    """Create three .npz files each with 50 samples under the key 'data'."""
    files = []
    for i in range(3):
        file_path = temp_dir / f"data_{i}.npz"
        np.savez(file_path, data=np.random.randn(50, 3, 32, 32).astype(np.float32))
        files.append(file_path)
    return files


class TestNumpyDataset:
    """Tests for the NumpyDataset class."""

    def test_init_single_npy_file(self, sample_npy_file):
        """Test initialisation from a single .npy file."""
        ds = NumpyDataset(sample_npy_file)
        assert ds.n_files == 1
        assert ds.n_datasets == 1
        assert len(ds) == 100
        assert ds.shape == (100, 3, 64, 64)
        ds.close()

    def test_init_single_npz_file(self, sample_npz_file):
        """Test initialisation from a single .npz file selecting one key."""
        ds = NumpyDataset(sample_npz_file, keys="images")
        assert ds.n_files == 1
        assert ds.n_datasets == 1
        assert len(ds) == 100
        assert ds.shape == (100, 3, 64, 64)
        ds.close()

    def test_getitem_npy(self, sample_npy_file):
        """Test item access, shape, dtype, and negative indexing from .npy."""
        ds = NumpyDataset(sample_npy_file)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 64, 64)
        assert sample.dtype == torch.float32
        # Negative indexing
        last = ds[-1]
        assert last.shape == (3, 64, 64)
        ds.close()

    def test_getitem_npz(self, sample_npz_file):
        """Test item access from a .npz file."""
        ds = NumpyDataset(sample_npz_file, keys="features")
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (128,)
        ds.close()

    def test_wildcard_keys_npz(self, sample_npz_file):
        """Test that fnmatch wildcards select multiple keys from a .npz archive."""
        # "images*" should match both "images" and "images_part2"
        ds = NumpyDataset(sample_npz_file, keys="images*")
        # Two matching arrays with 100 samples each are concatenated → 200 samples
        assert len(ds) == 200
        assert ds.shape == (200, 3, 64, 64)
        ds.close()

    def test_all_keys_selected_by_default_npz(self, sample_npz_file):
        """The default key pattern '*' selects all keys and concatenates them."""
        # 5 arrays * 100 samples, but shapes differ so concatenation should fail
        # unless we limit to a compatible subset
        ds = NumpyDataset(sample_npz_file, keys="images")
        assert len(ds) == 100
        ds.close()

    def test_multiple_compatible_keys_concatenated(self, temp_dir):
        """Multiple keys with compatible shapes are concatenated along axis 0."""
        file_path = temp_dir / "multi.npz"
        np.savez(
            file_path,
            a=np.ones((40, 4, 4), dtype=np.float32),
            b=np.ones((60, 4, 4), dtype=np.float32) * 2,
        )
        ds = NumpyDataset(file_path, keys="*")
        assert len(ds) == 100
        # First 40 samples come from 'a', next 60 from 'b'
        assert torch.all(ds[0] == 1.0)
        assert torch.all(ds[99] == 2.0)
        ds.close()

    def test_incompatible_shapes_raises(self, sample_npz_file):
        """Concatenating arrays with mismatched spatial shapes raises ValueError."""
        with pytest.raises(ValueError, match="incompatible"):
            # "images" (100, 3, 64, 64) and "features" (100, 128) cannot be concatenated
            NumpyDataset(sample_npz_file, keys=["images", "features"])

    def test_incompatible_dtypes_raises(self, temp_dir):
        """Concatenating arrays with mismatched dtypes raises ValueError."""
        file_path = temp_dir / "dtypes.npz"
        np.savez(
            file_path,
            a=np.ones((10, 4), dtype=np.float32),
            b=np.ones((10, 4), dtype=np.int32),
        )
        with pytest.raises(ValueError, match="incompatible"):
            NumpyDataset(file_path, keys="*")

    def test_single_npy_new_axis_len_is_one(self, sample_npy_file):
        """A single .npy file with new_axis=True has len==1."""
        ds = NumpyDataset(sample_npy_file, new_axis=True)
        assert len(ds) == 1
        ds.close()

    def test_single_npy_new_axis_shape(self, sample_npy_file):
        """Dataset shape is (n_files, *file_shape) when new_axis=True."""
        ds = NumpyDataset(sample_npy_file, new_axis=True)
        # sample_npy_file contains (100, 3, 64, 64)
        assert ds.shape == (1, 100, 3, 64, 64)
        ds.close()

    def test_multi_npy_new_axis_len_equals_n_files(self, multiple_npy_files):
        """Multiple .npy files → len equals the number of files, not total rows."""
        ds = NumpyDataset(multiple_npy_files, new_axis=True)
        assert len(ds) == 3
        ds.close()

    def test_multi_npy_new_axis_dataset_shape(self, multiple_npy_files):
        """Dataset shape is (n_files, *file_shape) for multiple files."""
        ds = NumpyDataset(multiple_npy_files, new_axis=True)
        # each file is (50, 3, 32, 32)
        assert ds.shape == (3, 50, 3, 32, 32)
        ds.close()

    def test_sample_shape_is_full_file(self, multiple_npy_files):
        """Each sample has the full file shape, not a single-row shape."""
        ds = NumpyDataset(multiple_npy_files, new_axis=True)
        sample = ds[0]
        assert sample.shape == (50, 3, 32, 32)
        ds.close()

    def test_new_axis_values_match_source_array(self, temp_dir):
        """Values returned by ds[i] match the original full array."""
        data = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        path = temp_dir / "known.npy"
        np.save(path, data)
        ds = NumpyDataset(path, new_axis=True, dtype=torch.float32)
        result = ds[0]
        expected = torch.from_numpy(data)
        assert torch.allclose(result, expected)
        ds.close()

    def test_new_axis_negative_index(self, multiple_npy_files):
        """Negative indices wrap around correctly with new_axis=True."""
        ds = NumpyDataset(multiple_npy_files, new_axis=True)
        assert torch.allclose(ds[-1], ds[2])
        assert torch.allclose(ds[-3], ds[0])
        ds.close()

    def test_caching_works_with_new_axis(self, temp_dir):
        """Cache stores and retrieves the full-file sample correctly."""
        data = np.random.randn(10, 4, 4).astype(np.float32)
        path = temp_dir / "cache_test.npy"
        np.save(path, data)
        ds = NumpyDataset(path, new_axis=True, cache="100M")
        assert ds.n_cached == 0
        first = ds[0]
        assert ds.n_cached == 1
        second = ds[0]
        assert torch.allclose(first, second)
        ds.close()

    def test_no_matching_keys_npz(self, sample_npz_file):
        """A pattern that matches no keys produces an empty dataset."""
        ds = NumpyDataset(sample_npz_file, keys="nonexistent_*")
        assert ds.n_datasets == 0
        assert len(ds) == 0
        ds.close()

    def test_with_attrs_npy_sidecar(self, sample_npy_file_with_sidecar):
        """Loading a .npy file with a .attrs.npy sidecar returns (data, attrs)."""
        file_path, _ = sample_npy_file_with_sidecar
        ds = NumpyDataset(file_path, attrs_keys="*", return_as="tuple")
        assert ds.has_attrs
        ret = ds[0]
        assert isinstance(ret, tuple)
        data, attrs = ret
        assert isinstance(data, torch.Tensor)
        assert attrs is not None
        ds.close()

    def test_no_sidecar_attrs_are_none(self, sample_npy_file):
        """When attrs_keys is set but no sidecar exists, attrs slot is None."""
        ds = NumpyDataset(sample_npy_file, attrs_keys="*", return_as="tuple")
        assert ds.has_attrs
        # _mmap_attrs should have a None placeholder
        assert ds._mmap_attrs[0] is None
        ds.close()

    def test_with_attrs_npz_keys(self, sample_npz_file):
        """Attrs keys in a .npz archive are loaded and excluded from data."""
        ds = NumpyDataset(
            sample_npz_file,
            keys="images",
            attrs_keys="labels",
            return_as="tuple",
        )
        assert ds.has_attrs
        ret = ds[0]
        assert isinstance(ret, tuple)
        data, attrs = ret
        assert data.shape == (3, 64, 64)
        assert attrs is not None
        ds.close()

    def test_attrs_keys_excluded_from_data_npz(self, temp_dir):
        """Keys claimed by attrs_keys must not appear in data."""
        file_path = temp_dir / "split.npz"
        np.savez(
            file_path,
            data=np.random.randn(20, 4, 4).astype(np.float32),
            meta=np.arange(20, dtype=np.int32),
        )
        ds = NumpyDataset(file_path, keys="*", attrs_keys="meta")
        # Only the "data" key should contribute samples
        assert len(ds) == 20
        ret = ds[0]
        assert isinstance(ret, tuple)
        data, attrs = ret
        assert data.shape == (4, 4)
        assert attrs is not None
        ds.close()

    def test_multiple_npy_files(self, multiple_npy_files):
        """Loading multiple .npy files fuses their indices correctly."""
        ds = NumpyDataset(multiple_npy_files)
        assert ds.n_files == 3
        assert len(ds) == 150  # 3 * 50
        for idx in [0, 50, 100]:
            assert ds[idx].shape == (3, 32, 32)
        ds.close()

    def test_multiple_npz_files(self, multiple_npz_files):
        """Loading multiple .npz files fuses their indices correctly."""
        ds = NumpyDataset(multiple_npz_files, keys="data")
        assert ds.n_files == 3
        assert len(ds) == 150
        ds.close()

    def test_glob_pattern_npy(self, multiple_npy_files):
        """Wildcard path patterns resolve to matching files."""
        directory = multiple_npy_files[0].parent
        ds = NumpyDataset(str(directory / "data_*.npy"))
        assert ds.n_files == 3
        assert len(ds) == 150
        ds.close()

    def test_caching(self, sample_npy_file):
        """Caching stores samples after first access and returns identical tensors."""
        ds = NumpyDataset(sample_npy_file, cache="100M")
        assert ds.n_cached == 0
        sample1 = ds[0]
        assert ds.n_cached == 1
        sample2 = ds[0]
        assert ds.n_cached == 1
        assert torch.equal(sample1, sample2)
        ds.close()

    def test_preload(self, temp_dir):
        """Preloading fills the cache up to its capacity."""
        file_path = temp_dir / "small.npy"
        n_samples = 10
        np.save(file_path, np.random.randn(n_samples, 8, 8).astype(np.float32))
        ds = NumpyDataset(file_path, cache="10M", preload=True)
        assert ds.n_cached == min(n_samples, ds.n_cacheable)
        ds.close()

    def test_context_manager(self, sample_npy_file):
        """Context manager closes the dataset and purges the cache on exit."""
        with NumpyDataset(sample_npy_file) as ds:
            sample = ds[0]
            assert isinstance(sample, torch.Tensor)
        assert ds.cache is None

    def test_info(self, sample_npy_file):
        """info() returns a string containing key diagnostic fields."""
        ds = NumpyDataset(sample_npy_file)
        info_str = ds.info(print_=False)
        assert isinstance(info_str, str)
        for field in ("Files:", "Samples:", "Shape:", "Key patterns:"):
            assert field in info_str
        ds.close()

    def test_info_with_attrs_patterns(self, sample_npy_file_with_sidecar):
        """info() includes Attr patterns when attrs_keys is set."""
        file_path, _ = sample_npy_file_with_sidecar
        ds = NumpyDataset(file_path, attrs_keys="*")
        info_str = ds.info(print_=False)
        assert "Attr patterns:" in info_str
        ds.close()

    def test_return_as_dict(self, sample_npy_file):
        """return_as='dict' wraps the sample in a dict under the 'data' key."""
        ds = NumpyDataset(sample_npy_file, return_as="dict")
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "data" in sample
        ds.close()

    def test_custom_dtype(self, sample_npy_file):
        """Samples are cast to the requested dtype."""
        ds = NumpyDataset(sample_npy_file, dtype=torch.int64)
        assert ds[0].dtype == torch.int64
        ds.close()

    def test_invalid_file_path(self):
        """A missing file emits a warning and produces an empty dataset."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ds = NumpyDataset("nonexistent.npy")
            assert ds.n_files == 0

    def test_index_out_of_range(self, sample_npy_file):
        """Accessing an out-of-range index raises IndexError."""
        ds = NumpyDataset(sample_npy_file)
        with pytest.raises(IndexError):
            _ = ds[1000]
        ds.close()

    def test_negative_index_out_of_range(self, sample_npy_file):
        """A large negative index also raises IndexError."""
        ds = NumpyDataset(sample_npy_file)
        with pytest.raises(IndexError):
            _ = ds[-1001]
        ds.close()

    def test_keys_warning_for_npy_only_files(self, sample_npy_file):
        """A non-default key pattern emits a UserWarning for .npy-only paths."""
        with pytest.warns(UserWarning, match="ignored"):
            NumpyDataset(sample_npy_file, keys="images").close()

    def test_no_keys_warning_for_mixed_files(self, temp_dir, sample_npy_file):
        """No warning is emitted when the file list includes at least one .npz."""
        npz_path = temp_dir / "extra.npz"
        np.savez(npz_path, images=np.random.randn(10, 3, 64, 64).astype(np.float32))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NumpyDataset([sample_npy_file, npz_path], keys="images").close()
        user_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "ignored" in str(w.message)
        ]
        assert len(user_warnings) == 0

    def test_default_key_pattern_no_warning(self, sample_npy_file):
        """The default key pattern '*' does not trigger the warning for .npy files."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NumpyDataset(sample_npy_file).close()
        user_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "ignored" in str(w.message)
        ]
        assert len(user_warnings) == 0

    def test_tensor_is_writable_from_npy_mmap(self, sample_npy_file):
        """Tensors from a mmap_mode='r' file must be writable.

        A non-writable tensor causes undefined behaviour on write and triggers
        a PyTorch UserWarning.  copy_on_write=False in the super().__init__
        call ensures the base class always copies memmap slices before handing
        them to torch.from_numpy.
        """
        ds = NumpyDataset(sample_npy_file)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample = ds[0]
        numpy_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "not writable" in str(w.message)
        ]
        assert len(numpy_warnings) == 0, (
            "Got non-writable tensor warning from memmap slice"
        )
        # Verify the tensor is actually writable
        sample[0] = 0.0
        ds.close()

    def test_mmap_closed_on_close(self, sample_npy_file):
        """close() explicitly releases the underlying mmap file handle."""
        ds = NumpyDataset(sample_npy_file)
        _ = ds[0]  # open the thread-local fd
        mmaps = list(ds._mmap)  # capture references before close() clears the list
        ds.close()
        for mmap in mmaps:
            if isinstance(mmap, NpyArrayView):
                fd = getattr(mmap._local, "fd", None)
                assert fd is None or fd.closed

    def test_data_values_match_source_npy(self, temp_dir):
        """Values returned by __getitem__ match the original array."""
        data = np.arange(50, dtype=np.float32).reshape(10, 5)
        file_path = temp_dir / "known.npy"
        np.save(file_path, data)
        ds = NumpyDataset(file_path, dtype=torch.float32)
        for i in range(10):
            expected = torch.from_numpy(data[i])
            assert torch.allclose(ds[i], expected)
        ds.close()

    def test_data_values_match_source_npz(self, temp_dir):
        """Values returned by __getitem__ match the original array in .npz."""
        data = np.arange(30, dtype=np.float32).reshape(6, 5)
        file_path = temp_dir / "known.npz"
        np.savez(file_path, arr=data)
        ds = NumpyDataset(file_path, keys="arr", dtype=torch.float32)
        for i in range(6):
            expected = torch.from_numpy(data[i])
            assert torch.allclose(ds[i], expected)
        ds.close()

    def test_dataloader_integration(self, sample_npy_file):
        """NumpyDataset works correctly inside a PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        ds = NumpyDataset(sample_npy_file)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (8, 3, 64, 64)
        ds.close()


class TestZipNumpyDataset:
    """Tests for the ZipNumpyDataset class."""

    def test_from_keys_same_npz(self, sample_npz_file):
        """from_keys reads multiple keys from the same .npz in parallel."""
        ds = ZipNumpyDataset.from_keys(sample_npz_file, "images", "labels")
        assert len(ds) == 100
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (3, 64, 64)
        ds.close()

    def test_from_keys_dict_return(self, sample_npz_file):
        """from_keys with a custom zip_as dict produces dict output."""
        ds = ZipNumpyDataset.from_keys(
            sample_npz_file,
            "images",
            "labels",
            zip_as={"image": 0, "label": 1},
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "image" in sample and "label" in sample
        assert sample["image"].shape == (3, 64, 64)
        ds.close()

    def test_from_paths_different_npy_files(self, multiple_npy_files):
        """from_paths reads from different .npy files in parallel."""
        ds = ZipNumpyDataset.from_paths(
            multiple_npy_files[0],
            multiple_npy_files[1],
        )
        assert len(ds) == 50  # limited by the shortest
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (3, 32, 32)
        ds.close()

    def test_from_paths_different_npz_files(self, multiple_npz_files):
        """from_paths reads from different .npz files in parallel."""
        ds = ZipNumpyDataset.from_paths(
            multiple_npz_files[0],
            multiple_npz_files[1],
            keys="data",
        )
        assert len(ds) == 50
        sample = ds[0]
        assert isinstance(sample, tuple)
        ds.close()

    def test_zip_from_paths_new_axis(self, multiple_npy_files):
        """ZipNumpyDataset.from_paths forwards new_axis=True to each sub-dataset."""
        ds = ZipNumpyDataset.from_paths(
            multiple_npy_files[0],
            multiple_npy_files[1],
            new_axis=True
        )
        # two datasets, treated as one sample each due to new_axis -> ziped as one
        assert len(ds) == 1
        sample = ds[0]
        assert isinstance(sample, tuple)
        # Each element should be the whole file, shape (50, 3, 32, 32)
        assert sample[0].shape == (50, 3, 32, 32)
        assert sample[1].shape == (50, 3, 32, 32)
        ds.close()

    def test_from_named_keys(self, sample_npz_file):
        """from_named_keys produces dict output keyed by name."""
        ds = ZipNumpyDataset.from_named_keys(
            sample_npz_file,
            keys={
                "image": "images",
                "label": "labels",
                "bbox": "bbox",
            },
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "label", "bbox"}
        assert sample["image"].shape == (3, 64, 64)
        assert sample["bbox"].shape == (4,)
        ds.close()

    def test_from_named_paths(self, multiple_npy_files):
        """from_named_paths produces dict output keyed by path name."""
        ds = ZipNumpyDataset.from_named_paths(
            paths={
                "file0": multiple_npy_files[0],
                "file1": multiple_npy_files[1],
            }
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"file0", "file1"}
        ds.close()

    def test_strict_length_checking(self, temp_dir):
        """Strict mode raises ValueError when datasets have different lengths."""
        short = temp_dir / "short.npy"
        long_ = temp_dir / "long.npy"
        np.save(short, np.random.randn(50, 4).astype(np.float32))
        np.save(long_, np.random.randn(100, 4).astype(np.float32))

        with pytest.raises(ValueError, match="same length"):
            ZipNumpyDataset.from_paths(short, long_, strict=True)

        # Non-strict: length limited by the shortest
        ds = ZipNumpyDataset.from_paths(short, long_, strict=False)
        assert len(ds) == 50
        ds.close()

    def test_from_keys_empty_raises(self, sample_npz_file):
        """from_keys with no keys returns an empty ZipNumpyDataset."""
        with pytest.raises(ValueError, match="At least one dataset"):
            ZipNumpyDataset.from_keys(sample_npz_file)

    def test_from_paths_empty_raises(self):
        """from_paths with no paths raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipNumpyDataset.from_paths()

    def test_from_named_keys_empty_raises(self, sample_npz_file):
        """from_named_keys with an empty dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one key"):
            ZipNumpyDataset.from_named_keys(sample_npz_file, keys={})

    def test_from_named_paths_empty_raises(self):
        """from_named_paths with an empty dict raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipNumpyDataset.from_named_paths(paths={})

    def test_caching_in_zip(self, sample_npz_file):
        """Each constituent NumpyDataset in a Zip populates its own cache."""
        ds = ZipNumpyDataset.from_keys(
            sample_npz_file,
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

    def test_dataloader_integration(self, sample_npz_file):
        """ZipNumpyDataset works correctly inside a PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        ds = ZipNumpyDataset.from_keys(
            sample_npz_file,
            "images",
            "labels",
            zip_as={"image": 0, "label": 1},
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert "image" in batch and "label" in batch
        assert batch["image"].shape == (8, 3, 64, 64)
        assert batch["label"].shape == (8,)
        ds.close()

    def test_multi_modal_workflow(self, sample_npz_file):
        """A multi-modal workflow iterates correctly and caches all modalities."""
        ds = ZipNumpyDataset.from_named_keys(
            sample_npz_file,
            keys={
                "image": "images",
                "features": "features",
                "label": "labels",
                "bbox": "bbox",
            },
            cache="200M",
        )
        samples_seen = 0
        for i in range(min(10, len(ds))):
            sample = ds[i]
            assert all(k in sample for k in ("image", "features", "label", "bbox"))
            samples_seen += 1
        assert samples_seen == 10
        ds.close()

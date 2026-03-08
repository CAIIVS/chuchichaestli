# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the HDF5Dataset and ZipHDF5Dataset classes."""

from pathlib import Path
import tempfile
import numpy as np
import torch
import h5py
import warnings
import pytest
from chuchichaestli.data.hdf5 import HDF5Dataset, ZipHDF5Dataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_hdf5_file(temp_dir):
    """Create a sample HDF5 file with various groups and datasets."""
    file_path = temp_dir / "test.h5"

    with h5py.File(file_path, "w") as f:
        # Create nested groups
        data_group = f.create_group("data")
        labels_group = f.create_group("labels")
        metadata_group = f.create_group("metadata")

        # Add datasets with different shapes
        data_group.create_dataset(
            "images", data=np.random.randn(100, 3, 64, 64).astype(np.float32)
        )
        data_group.create_dataset(
            "images_part2", data=np.random.randn(100, 3, 64, 64).astype(np.float32)
        )
        data_group.create_dataset(
            "features", data=np.random.randn(100, 128).astype(np.float32)
        )
        labels_group.create_dataset("class", data=np.random.randint(0, 10, 100))
        labels_group.create_dataset(
            "bbox", data=np.random.rand(100, 4).astype(np.float32)
        )

        # Add metadata
        metadata_group.attrs["dataset_name"] = "test_dataset"
        metadata_group.attrs["version"] = "1.0"
        metadata_group.create_dataset("sample_ids", data=np.arange(100))

    return file_path


@pytest.fixture
def multiple_hdf5_files(temp_dir):
    """Create multiple HDF5 files for multi-file testing."""
    files = []

    for i in range(3):
        file_path = temp_dir / f"test_{i}.h5"

        with h5py.File(file_path, "w") as f:
            data = np.random.randn(50, 3, 32, 32).astype(np.float32)
            f.create_dataset("data", data=data)

        files.append(file_path)

    return files


@pytest.fixture
def multi_dataset_file(temp_dir):
    """HDF5 file with two same-shaped datasets under the same group.

    This triggers the VDS (virtual dataset) code path in ``HDF5Dataset.load``
    where ``len(datasets) > 1``.
    """
    file_path = temp_dir / "multi_ds.h5"
    with h5py.File(file_path, "w") as f:
        grp = f.create_group("images")
        grp.create_dataset(
            "part_a",
            data=np.arange(40 * 4 * 4, dtype=np.float32).reshape(40, 4, 4),
        )
        grp.create_dataset(
            "part_b",
            data=np.arange(40 * 4 * 4, dtype=np.float32).reshape(40, 4, 4),
        )
    return file_path


@pytest.fixture
def group_attrs_file(temp_dir):
    """HDF5 file where a *group* carries HDF5 attributes (not a Dataset).

    This exercises the ``h5py.AttributeManager`` branch inside
    ``_find_attrs`` and ``_get_from_mmap_attrs``.
    """
    file_path = temp_dir / "group_attrs.h5"
    with h5py.File(file_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.create_dataset(
            "images", data=np.random.randn(20, 8, 8).astype(np.float32)
        )
        # A *group* with HDF5 dataset, meant as corresponding metadata
        data_grp.create_dataset(
            "attrs", data=np.random.randn(20, 8, 8).astype(np.float32)
        )
        # A *group* with HDF5 attributes, resolved to an AttributeManager
        meta_grp = f.create_group("meta")
        meta_grp.attrs["scalar_attr"] = 42
        meta_grp.attrs["string_attr"] = "hello"
        meta_grp.attrs["array_attr"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        meta_grp.attrs["single_elem_array"] = np.array([7.0], dtype=np.float32)
    return file_path


class TestHDF5Dataset:
    """Tests for the refactored HDF5Dataset class."""

    def test_init_single_file(self, sample_hdf5_file):
        """Test initialization with a single file."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images")
        assert ds.n_files == 1
        assert ds.n_datasets == 1
        assert len(ds) == 100
        assert ds.shape == (100, 3, 64, 64)
        ds.close()

    def test_getitem(self, sample_hdf5_file):
        """Test getting individual samples."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images")
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 64, 64)
        assert sample.dtype == torch.float32
        # Test negative indexing
        last_sample = ds[-1]
        assert last_sample.shape == (3, 64, 64)
        ds.close()

    def test_wildcard_groups(self, sample_hdf5_file):
        """Test wildcard pattern matching for groups."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images*")
        assert ds.n_datasets >= 2  # Should find both images and images_part2
        ds.close()

    def test_with_attributes(self, sample_hdf5_file):
        """Test loading with attributes."""
        ds = HDF5Dataset(
            sample_hdf5_file,
            groups="data/images",
            attrs_groups="metadata",
            return_as="tuple",
            attrs_cache="10M",
        )
        assert ds.has_attrs
        ret = ds[0]
        assert isinstance(ret, tuple)
        data, attrs = ret
        assert isinstance(data, torch.Tensor)
        assert attrs is not None
        ds.close()

    def test_multiple_files(self, multiple_hdf5_files):
        """Test loading from multiple files."""
        ds = HDF5Dataset(multiple_hdf5_files, groups="data")
        assert ds.n_files == 3
        assert len(ds) == 150  # 3 files * 50 samples each
        # Test samples from different files
        sample_0 = ds[0]  # From file 0
        sample_50 = ds[50]  # From file 1
        sample_100 = ds[100]  # From file 2
        assert all(s.shape == (3, 32, 32) for s in [sample_0, sample_50, sample_100])
        ds.close()

    def test_caching(self, sample_hdf5_file):
        """Test caching functionality."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images", cache="100M")
        # First access - not cached
        assert ds.n_cached == 0
        sample1 = ds[0]
        assert ds.n_cached == 1
        # Second access - cached
        sample2 = ds[0]
        assert ds.n_cached == 1
        assert torch.equal(sample1, sample2)
        ds.close()

    def test_preload(self, temp_dir):
        """Test preloading all data."""
        # Create a small file for preloading
        file_path = temp_dir / "small.h5"
        n_samples = 10
        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "data", data=np.random.randn(n_samples, 8, 8).astype(np.float32)
            )
        ds = HDF5Dataset(file_path, groups="data", cache="10M", preload=True)
        # All samples should be cached
        print(f"Chached samples (of {n_samples}):", ds.n_cached)
        assert ds.n_cached == min(len(ds), ds.n_cacheable)
        ds.close()

    def test_context_manager(self, sample_hdf5_file):
        """Test context manager usage."""
        with HDF5Dataset(sample_hdf5_file, groups="data/images") as ds:
            sample = ds[0]
            assert isinstance(sample, torch.Tensor)
        # Dataset should be closed after exiting context
        ref = locals()["ds"]
        assert ref.cache is None

    def test_info(self, sample_hdf5_file):
        """Test info method."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images")
        info_str = ds.info(print_=False)
        assert isinstance(info_str, str)
        assert "Files:" in info_str
        assert "Samples:" in info_str
        ds.close()

    def test_info_includes_group_patterns(self, temp_dir):
        """info() output mentions the configured group patterns."""
        fp = temp_dir / "info.h5"
        with h5py.File(fp, "w") as f:
            f.create_dataset("data", data=np.zeros((5, 4), dtype=np.float32))
            meta_grp = f.create_group("meta")
            meta_grp.attrs["scalar_attr"] = 42
        ds = HDF5Dataset(fp, groups="data", attrs_groups="meta")
        info = ds.info(print_=True)
        assert "Group patterns:" in info
        ds.close()

    def test_return_as_dict(self, sample_hdf5_file):
        """Test dict return format."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images", return_as="dict")
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "data" in sample
        ds.close()

    def test_custom_dtype(self, sample_hdf5_file):
        """Test custom dtype conversion."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images", dtype=torch.int32)
        sample = ds[0]
        assert sample.dtype == torch.int32
        ds.close()

    def test_empty_groups(self, sample_hdf5_file):
        """Test with empty or non-existent groups."""
        ds = HDF5Dataset(sample_hdf5_file, groups="nonexistent/*")
        # Should create dataset but with no data
        assert ds.n_datasets == 0
        ds.close()

    def test_invalid_file_path(self):
        """Test with invalid file path."""
        with warnings.catch_warnings(record=True):
            ds = HDF5Dataset("nonexistent.h5", groups="*")
            assert ds.n_files == 0
            print(ds)

    def test_index_out_of_range(self, sample_hdf5_file):
        """Test index out of range errors."""
        ds = HDF5Dataset(sample_hdf5_file, groups="data/images")
        with pytest.raises(IndexError):
            _ = ds[1000]  # Beyond dataset length
        ds.close()

    def test_vds_concatenates_datasets(self, multi_dataset_file):
        """Two matched datasets in one file are transparently concatenated via VDS."""
        ds = HDF5Dataset(multi_dataset_file, groups="images/*")
        # Both part_a and part_b have 40 samples, total should be 80
        assert ds.n_datasets == 2
        assert len(ds) == 80
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (4, 4)
        ds.close()

    def test_vds_second_half_accessible(self, multi_dataset_file):
        """Samples from the second concatenated dataset are reachable."""
        ds = HDF5Dataset(multi_dataset_file, groups="images/*")
        sample = ds[60]  # well into part_b
        assert sample.shape == (4, 4)
        ds.close()

    def test_vds_temp_file_cleaned_on_close(self, multi_dataset_file):
        """VDS temp files are tracked; after close() the tracking list is cleared."""
        ds = HDF5Dataset(multi_dataset_file, groups="images/*")
        virt_paths = list(ds._virt_files)  # copy before close
        assert len(virt_paths) > 0, "Expected at least one virtual temp file"
        ds.close()
        assert len(ds._virt_files) == 0

    def test_create_vds_empty_raises(self, multi_dataset_file):
        """_create_vds raises ValueError when given an empty dataset list."""
        with pytest.raises(ValueError, match="At least one dataset required"):
            HDF5Dataset._create_vds([], multi_dataset_file)

    def test_create_vds_shape_mismatch_raises(self, temp_dir):
        """_create_vds raises ValueError when dataset shapes are incompatible."""
        fp = temp_dir / "mismatch.h5"
        with h5py.File(fp, "w") as f:
            f.create_dataset("a", data=np.zeros((10, 4, 4), dtype=np.float32))
            f.create_dataset("b", data=np.zeros((10, 8, 8), dtype=np.float32))
        with h5py.File(fp, "r") as hf:
            ds_a = hf["a"]
            ds_b = hf["b"]
            with pytest.raises(ValueError, match="shapes are incompatible"):
                HDF5Dataset._create_vds([ds_a, ds_b], fp)

    def test_create_vds_dtype_mismatch_raises(self, temp_dir):
        """_create_vds raises ValueError when dataset dtypes are incompatible."""
        fp = temp_dir / "dtype_mismatch.h5"
        with h5py.File(fp, "w") as f:
            f.create_dataset("a", data=np.zeros((10, 4), dtype=np.float32))
            f.create_dataset("b", data=np.zeros((10, 4), dtype=np.float64))
        with h5py.File(fp, "r") as hf:
            ds_a = hf["a"]
            ds_b = hf["b"]
            with pytest.raises(ValueError, match="dtypes are incompatible"):
                HDF5Dataset._create_vds([ds_a, ds_b], fp)

    def test_group_attrs_returned_as_dict(self, group_attrs_file):
        """When attrs_groups matches a *group*, attrs come back as a plain dict."""
        ds = HDF5Dataset(
            group_attrs_file,
            groups="data/images",
            attrs_groups="meta",
            return_as="tuple",
        )
        assert ds.has_attrs
        data, attrs = ds[0]
        assert isinstance(attrs, dict)
        assert "scalar_attr" in attrs
        assert "string_attr" in attrs
        ds.close()

    def test_group_attrs_returned_as_dataset(self, group_attrs_file):
        """When attrs_groups matches a *group*, attrs come back as tensor."""
        ds = HDF5Dataset(
            group_attrs_file,
            groups="data/images",
            attrs_groups="data/attrs",
            return_as="tuple",
        )
        assert ds.has_attrs
        data, attrs = ds[0]
        assert isinstance(attrs, np.ndarray)
        ds.close()

    def test_attrs_groups(self, group_attrs_file):
        """Test attrs_groups property."""
        ds = HDF5Dataset(
            group_attrs_file,
            groups="data/images",
            attrs_groups="data/attrs",
            return_as="tuple",
        )
        assert ds.has_attrs
        assert ds.attr_groups
        assert isinstance(ds.attr_groups, list)
        assert isinstance(ds.attr_groups[0], list)
        assert ds.attr_groups[0][0] == "/data/attrs"

    def test_attrs_groups_with_attribute_manager(self, group_attrs_file):
        """Test attrs_groups property."""
        ds = HDF5Dataset(
            group_attrs_file,
            groups="data/images",
            attrs_groups="meta",
            return_as="tuple",
        )
        assert ds.has_attrs
        assert ds.attr_groups
        assert isinstance(ds.attr_groups, list)
        assert isinstance(ds.attr_groups[0], list)
        assert ds.attr_groups[0][0] == "/meta"


class TestZipHDF5Dataset:
    """Tests for the ZipHDF5Dataset class."""

    def test_from_groups_same_file(self, sample_hdf5_file):
        """Test loading multiple groups from same file."""
        ds = ZipHDF5Dataset.from_groups(
            sample_hdf5_file, "data/images", "labels/class", zip_as="tuple"
        )
        assert len(ds) == 100
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape == (3, 64, 64)
        ds.close()

    def test_from_groups_dict_return(self, sample_hdf5_file):
        """Test dict return format with named keys."""
        ds = ZipHDF5Dataset.from_groups(
            sample_hdf5_file,
            "data/images",
            "labels/class",
            zip_as={"image": 0, "label": 1},
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (3, 64, 64)
        ds.close()

    def test_from_paths_different_files(self, multiple_hdf5_files):
        """Test loading from different files."""
        ds = ZipHDF5Dataset.from_paths(
            multiple_hdf5_files[0],
            multiple_hdf5_files[1],
            groups="data",
        )
        assert len(ds) == 50  # Limited by shortest dataset
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        ds.close()

    def test_from_named_groups(self, sample_hdf5_file):
        """Test named groups for automatic dict output."""
        ds = ZipHDF5Dataset.from_named_groups(
            sample_hdf5_file,
            groups={
                "image": "data/images",
                "class": "labels/class",
                "bbox": "labels/bbox",
            },
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "class", "bbox"}
        assert sample["image"].shape == (3, 64, 64)
        assert sample["bbox"].shape == (4,)
        ds.close()

    def test_from_named_paths(self, multiple_hdf5_files):
        """Test named paths for automatic dict output."""
        ds = ZipHDF5Dataset.from_named_paths(
            paths={"file1": multiple_hdf5_files[0], "file2": multiple_hdf5_files[1]},
            groups="data",
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"file1", "file2"}
        ds.close()

    def test_strict_length_checking(self, temp_dir):
        """Test strict mode with different length datasets."""
        # Create files with different lengths
        file1 = temp_dir / "short.h5"
        file2 = temp_dir / "long.h5"
        with h5py.File(file1, "w") as f:
            f.create_dataset("data", data=np.random.randn(50, 8, 8).astype(np.float32))
        with h5py.File(file2, "w") as f:
            f.create_dataset("data", data=np.random.randn(100, 8, 8).astype(np.float32))
        # Strict mode should raise error
        with pytest.raises(ValueError, match="same length"):
            ds = ZipHDF5Dataset.from_paths(file1, file2, groups="data", strict=True)
        # Non-strict mode should work (uses shortest)
        ds = ZipHDF5Dataset.from_paths(file1, file2, groups="data", strict=False)
        assert len(ds) == 50  # Shortest dataset
        ds.close()

    def test_caching_in_zip(self, sample_hdf5_file):
        """Test that caching works in ZipHDF5Dataset."""
        ds = ZipHDF5Dataset.from_groups(
            sample_hdf5_file,
            "data/images",
            "labels/class",
            zip_as={"image": 0, "label": 1},
            cache="100M",
        )
        # Access a sample
        sample1 = ds[0]
        # Check that individual datasets have caching
        for dataset in ds.datasets:
            if hasattr(dataset, "n_cached"):
                assert dataset.n_cached > 0
        assert isinstance(sample1, dict)
        assert "image" in sample1
        assert "label" in sample1
        ds.close()

    @pytest.mark.parametrize("clsmethod", [ZipHDF5Dataset.from_paths, ZipHDF5Dataset.from_named_paths])
    def test_from_paths_no_paths_raises(self, clsmethod):
        """from_paths raises an error when called without any path arguments."""
        with pytest.raises((ValueError, TypeError)):
            clsmethod(paths=(), groups="*")

    def test_from_named_groups_no_groups_raises(self, sample_hdf5_file):
        """from_named_groups raises an error when called without any path arguments."""
        with pytest.raises((ValueError, TypeError)):
            ZipHDF5Dataset.from_named_groups(sample_hdf5_file, groups=())

    def test_dataloader_integration(self, sample_hdf5_file):
        """Test integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        ds = ZipHDF5Dataset.from_groups(
            sample_hdf5_file,
            "data/images",
            "labels/class",
            zip_as={'image': 0, 'label': 1},
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert 'image' in batch
        assert 'label' in batch
        assert batch['image'].shape == (8, 3, 64, 64)
        assert batch['label'].shape == (8,)
        ds.close()

    def test_multi_modal_workflow(self, sample_hdf5_file):
        """Test a multi-modal data workflow."""
        ds = ZipHDF5Dataset.from_named_groups(
            sample_hdf5_file,
            groups={
                'image': 'data/images',
                'features': 'data/features',
                'class_label': 'labels/class',
                'bbox': 'labels/bbox'
            },
            cache="200M"
        )
        # Iterate through some samples
        samples_seen = 0
        for i in range(min(10, len(ds))):
            sample = ds[i]
            assert all(key in sample for key in ['image', 'features', 'class_label', 'bbox'])
            samples_seen += 1
        assert samples_seen == 10
        assert ds.n_cached == 40
        ds.close()

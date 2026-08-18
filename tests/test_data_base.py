# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the data base module."""

import pytest
import warnings
import torch
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
from chuchichaestli.data.base import FileDataset, CachingDataset
from chuchichaestli.data.cache import nbytes


class DummyFileDataset(FileDataset):
    """Concrete implementation of FileDataset for testing."""

    FILE_EXTENSIONS = [".npy"]

    def load(self, **kwargs):
        """Load numpy files into memory maps."""
        for file_path in self.files:
            data = np.load(file_path, mmap_mode="r")
            self._mmap.append(data)
            # Load mmap for metadata if they exist
            attrs_path = file_path.with_suffix(".attrs.npy")
            if attrs_path.exists():
                attrs = np.load(attrs_path, allow_pickle=True)
                self._mmap_attrs.append(attrs)
        self._build_index()

    def close(self, **kwargs):
        """Close memory maps."""
        self._mmap = []
        self._mmap_attrs = []
        self._file_offsets = []


class DummyCachingDataset(CachingDataset):
    """Concrete implementation of CachingDataset for testing."""

    FILE_EXTENSIONS = [".npy"]

    def load(self, **kwargs):
        """Load numpy files into memory maps."""
        for file_path in self.files:
            data = np.load(file_path, mmap_mode="r")
            self._mmap.append(data)
            # Load attributes if they exist
            attrs_path = file_path.with_suffix(".attrs.npy")
            if attrs_path.exists():
                attrs = np.load(attrs_path, allow_pickle=True)
                self._mmap_attrs.append(attrs)
        self._build_index()

    def close(self, **kwargs):
        """Close memory maps and purge cache."""
        super().close()
        self._mmap = []
        self._mmap_attrs = []
        self._file_offsets = []

def filter_attrs_files(files):
    """Filter out .attrs.npy files from a list of paths.
    
    Args:
        files: List of file paths (strings or Path objects).
        
    Returns:
        List of paths excluding .attrs.npy files.
    """
    return [f for f in files if '.attrs' not in str(f)]

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return {
        'data1': np.random.randn(10, 5, 5).astype(np.float32),
        'data2': np.random.randn(15, 5, 5).astype(np.float32),
        'data3': np.random.randn(8, 5, 5).astype(np.float32),
    }

@pytest.fixture
def sample_attrs():
    """Generate sample attributes for testing."""
    return {
        'data1': np.array([{'id': i, 'label': f'sample_{i}'} for i in range(10)]),
        'data2': np.array([{'id': i, 'label': f'sample_{i}'} for i in range(15)]),
        'data3': np.array([{'id': i, 'label': f'sample_{i}'} for i in range(8)]),
    }

@pytest.fixture
def create_test_files(temp_dir, sample_data, sample_attrs):
    """Create test .npy files in the temporary directory."""
    files = []    
    for name, data in sample_data.items():
        file_path = temp_dir / f"{name}.npy"
        np.save(file_path, data)
        files.append(file_path)
        
        # Save corresponding attributes
        if name in sample_attrs:
            attrs_path = temp_dir / f"{name}.attrs.npy"
            np.save(attrs_path, sample_attrs[name])
    
    return filter_attrs_files(files)

@pytest.fixture
def create_nested_files(temp_dir):
    """Create nested directory structure with test files."""
    # Create subdirectories
    (temp_dir / "subdir1").mkdir()
    (temp_dir / "subdir2").mkdir()
    (temp_dir / "subdir1" / "nested").mkdir()
    
    files = []
    
    # Create files in different locations
    for i, path in enumerate([
        temp_dir / "file1.npy",
        temp_dir / "subdir1" / "file2.npy",
        temp_dir / "subdir1" / "nested" / "file3.npy",
        temp_dir / "subdir2" / "file4.npy",
    ]):
        data = np.random.randn(5, 3, 3).astype(np.float32)
        np.save(path, data)
        files.append(path)
    return filter_attrs_files(files)



class TestFileDataset:
    """Test static methods of FileDataset."""

    def test_split_glob_simple_path(self):
        """Test splitting a simple path without wildcards."""
        roots, patterns = FileDataset._split_glob("/path/to/file.txt")
        assert roots == ["/path/to/file.txt"]
        assert patterns == [None]

    def test_split_glob_with_wildcard(self):
        """Test splitting a path with wildcard."""
        roots, patterns = FileDataset._split_glob("/path/to/*.txt")
        assert roots == ["/path/to"]
        assert patterns == ["*.txt"]

    def test_split_glob_nested_wildcard(self):
        """Test splitting a path with nested wildcard."""
        roots, patterns = FileDataset._split_glob("/path/*/subdir/*.txt")
        assert roots == ["/path"]
        assert patterns == ["*/subdir/*.txt"]

    def test_split_glob_relative_path(self):
        """Test splitting with relative path option."""
        roots, patterns = FileDataset._split_glob("/path/to/*.txt", relative=True)
        assert roots == ["path/to"]
        assert patterns == ["*.txt"]

    def test_split_glob_list_of_paths(self):
        """Test splitting a list of paths."""
        paths = ["/path1/*.txt", "/path2/file.txt"]
        roots, patterns = FileDataset._split_glob(paths)
        assert len(roots) == 2
        assert len(patterns) == 2
        assert "/path1" in roots
        assert "*.txt" in patterns
        assert "/path2/file.txt" in roots
        assert None in patterns

    def test_glob_path_none(self):
        """Test glob_path with None input."""
        result = FileDataset.glob_path(None)
        assert result == []

    def test_glob_path_single_file(self, create_test_files):
        """Test glob_path with a single file."""
        file_path = create_test_files[0]
        result = FileDataset.glob_path(str(file_path))
        assert len(result) == 1
        assert result[0] == file_path

    def test_glob_path_wildcard(self, temp_dir, create_test_files):
        """Test glob_path with wildcard pattern."""
        pattern = str(temp_dir / "*.npy")
        result = FileDataset.glob_path(pattern, extensions=[".npy"])
        assert len(result) == 6  # 3+3 .npy files (from fixture create_test_files)

    def test_glob_path_with_extensions_filter(self, temp_dir, create_test_files):
        """Test glob_path with extension filtering."""
        # Create a .txt file that should be filtered out
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test")
        pattern = str(temp_dir / "data*")
        result = FileDataset.glob_path(pattern, extensions=[".npy"])
        # Should only find .npy files, not .txt or .attrs.npy
        assert all(f.suffix == ".npy" for f in result)

    def test_glob_path_recursive(self, create_nested_files, temp_dir):
        """Test glob_path with recursive wildcard."""
        pattern = str(temp_dir / "**" / "*.npy")
        result = FileDataset.glob_path(pattern, extensions=[".npy"])
        assert len(result) == 4  # Should find all nested files

    def test_glob_path_nonexistent_warns(self, temp_dir):
        """Test that glob_path warns about missing files."""
        pattern = str(temp_dir / "nonexistent" / "*.npy")
      
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = FileDataset.glob_path(pattern)
            assert len(result) == 0

    def test_check_files_exist(self, temp_dir, create_test_files):
        """Test check_files_exist method."""
        existing_file = create_test_files[0]
        missing_file = temp_dir / "missing.npy"
      
        files = [existing_file, missing_file]
        existing, missing = FileDataset.check_files_exist(files)
      
        assert len(existing) == 1
        assert existing[0] == existing_file
        assert len(missing) == 1
        assert missing[0] == missing_file

    def test_validate_files_success(self, create_test_files):
        """Test validate_files with valid files."""
        result = FileDataset.validate_files(
            create_test_files,
            extensions=[".npy"],
            check_exists=True,
            raise_on_error=True
        )
        assert len(result) == 3

    def test_validate_files_missing_raises(self, temp_dir):
        """Test validate_files raises on missing file."""
        missing_file = temp_dir / "missing.npy"
        with pytest.raises(FileNotFoundError, match="File not found"):
            FileDataset.validate_files(
                [missing_file],
                extensions=[".npy"],
                check_exists=True,
                raise_on_error=True
            )

    def test_validate_files_missing_warns(self, temp_dir):
        """Test validate_files warns on missing file when not raising."""
        missing_file = temp_dir / "missing.npy"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FileDataset.validate_files(
                [missing_file],
                extensions=[".npy"],
                check_exists=True,
                raise_on_error=False
            )
            assert len(result) == 0
            assert len(w) == 1
            assert "not found" in str(w[0].message)

    def test_validate_files_invalid_extension_raises(self, temp_dir):
        """Test validate_files raises on invalid extension."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test")
        with pytest.raises(ValueError, match="Invalid file extension"):
            FileDataset.validate_files(
                [txt_file],
                extensions=[".npy"],
                check_exists=False,
                raise_on_error=True
            )

    def test_validate_files_invalid_extension_warns(self, temp_dir):
        """Test validate_files warns on invalid extension when not raising."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FileDataset.validate_files(
                [txt_file],
                extensions=[".npy"],
                check_exists=False,
                raise_on_error=False
            )
            assert len(result) == 0
            assert len(w) == 1
            assert "Invalid extension" in str(w[0].message)

    def test_init_with_no_path(self):
        """Test initialization with no path."""
        dataset = DummyFileDataset(path=None)
        assert dataset.n_files == 0
        assert not dataset.has_files
        assert len(dataset._mmap) == 0

    def test_init_with_single_file(self, create_test_files):
        """Test initialization with a single file."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        assert dataset.n_files == 1
        assert dataset.has_files

    def test_init_with_multiple_files(self, temp_dir, create_test_files):
        """Test initialization with wildcard matching multiple files."""
        dataset = DummyFileDataset(path=create_test_files)
        assert dataset.n_files == 3

    def test_init_with_list_of_files(self, create_test_files):
        """Test initialization with list of file str paths."""
        paths = [str(f) for f in create_test_files]
        dataset = DummyFileDataset(path=paths)
        assert dataset.n_files == 3

    def test_dtype_setting(self, create_test_files):
        """Test that dtype is correctly set."""
        dataset = DummyFileDataset(
            path=str(create_test_files[0]),
            dtype=torch.float64
        )
        assert dataset.dtype == torch.float64

    def test_return_as_tuple(self, create_test_files):
        """Test return_as='tuple' setting."""
        dataset = DummyFileDataset(
            path=str(create_test_files[0]),
            return_as='tuple'
        )
        assert dataset.return_as == 'tuple'

    def test_return_as_dict(self, create_test_files):
        """Test return_as='dict' setting."""
        dataset = DummyFileDataset(
            path=str(create_test_files[0]),
            return_as='dict'
        )
        assert dataset.return_as == 'dict'

    def test_shape_property(self, temp_dir, create_test_files):
        """Test shape property."""
        dataset = DummyFileDataset(path=create_test_files)
        # Total samples = 10 + 15 + 8 = 33
        # Sample shape = (5, 5)
        assert dataset.shape == (33, 5, 5)

    def test_n_samples_property(self, temp_dir, create_test_files):
        """Test n_samples property."""
        dataset = DummyFileDataset(path=create_test_files)
        assert dataset.n_samples == 33

    def test_sample_shape_property(self, temp_dir, create_test_files):
        """Test sample_shape property."""
        dataset = DummyFileDataset(path=create_test_files)
        assert dataset.sample_shape == (5, 5)

    def test_len(self, temp_dir, create_test_files):
        """Test __len__ method."""
        dataset = DummyFileDataset(path=create_test_files)
        assert len(dataset) == 33

    def test_str_representation(self, create_test_files):
        """Test __str__ method."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        str_repr = str(dataset)
        assert "DummyFileDataset" in str_repr
        assert "#f1" in str_repr  # 1 file
        assert "#s10" in str_repr  # 10 samples
        print(str_repr)

    def test_repr(self, create_test_files):
        """Test __repr__ method."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        assert repr(dataset) == str(dataset)
        print([str(dataset)])

    def test_build_index(self, temp_dir, create_test_files):
        """Test _build_index method."""
        dataset = DummyFileDataset(path=create_test_files)
        # Offsets should be [0, 10, 25, 33] for files with 10, 15, 8 samples
        assert len(dataset._file_offsets) == 4
        assert dataset._file_offsets[0] == 0
        assert dataset._file_offsets[-1] == 33

    def test_map_index_first_file(self, temp_dir, create_test_files):
        """Test _map_index for indices in first file."""
        dataset = DummyFileDataset(path=create_test_files)
        file_idx, local_idx = dataset._map_index(0)
        assert local_idx == 0

    def test_map_index_middle_file(self, temp_dir, create_test_files):
        """Test _map_index for indices in middle file."""
        dataset = DummyFileDataset(path=create_test_files)
        # Index 15 should be in the second file (after 10 or 8 samples)
        file_idx, local_idx = dataset._map_index(15)
        assert file_idx >= 0 and file_idx < 3

    def test_map_index_out_of_range(self, temp_dir, create_test_files):
        """Test _map_index raises IndexError for out of range index."""
        dataset = DummyFileDataset(path=create_test_files)
        with pytest.raises(IndexError, match="out of range"):
            dataset._map_index(100)

    def test_map_index_out_of_range_new_axis(self, temp_dir, create_test_files):
        """Test _map_index raises IndexError for out of range index."""
        dataset = DummyFileDataset(path=create_test_files, new_axis=True)
        with pytest.raises(IndexError, match="out of range"):
            dataset._map_index(100)

    def test_getitem_positive_index(self, create_test_files):
        """Test __getitem__ with positive index."""
        dataset = DummyFileDataset(path=str(create_test_files[0]), has_attrs=False)
        print(dataset)
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5, 5)

    def test_getitem_negative_index(self, create_test_files):
        """Test __getitem__ with negative index."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        item = dataset[-1]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5, 5)

    def test_getitem_out_of_range_positive(self, create_test_files):
        """Test __getitem__ raises IndexError for positive out of range."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        with pytest.raises(IndexError, match="out of range"):
            _ = dataset[100]

    def test_getitem_out_of_range_negative(self, create_test_files):
        """Test __getitem__ raises IndexError for negative out of range."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        with pytest.raises(IndexError, match="out of range"):
            _ = dataset[-100]

    def test_format_output_no_attrs(self, create_test_files):
        """Test _format_output with no attributes."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        item = torch.randn(5, 5)
        result = dataset._format_output(item, None)
        assert torch.equal(result, item)

    def test_format_output_tuple(self, create_test_files):
        """Test _format_output with return_as='tuple'."""
        dataset = DummyFileDataset(
            path=str(create_test_files[0]),
            return_as='tuple'
        )
        item = torch.randn(5, 5)
        attrs = {'id': 0}
        result = dataset._format_output(item, attrs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], item)
        assert result[1] == attrs

    def test_format_output_dict(self, create_test_files):
        """Test _format_output with return_as='dict'."""
        dataset = DummyFileDataset(
            path=str(create_test_files[0]),
            return_as='dict'
        )
        item = torch.randn(5, 5)
        attrs = {'id': 0}
        result = dataset._format_output(item, attrs)
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'attrs' in result
        assert torch.equal(result['data'], item)
        assert result['attrs'] == attrs

    def test_format_output_dict_template(self, create_test_files):
        """Test _format_output with custom dict template."""
        dataset = DummyFileDataset(
            path=str(create_test_files[0]),
            return_as={'image': None, 'metadata': None}
        )
        item = torch.randn(5, 5)
        attrs = {'id': 0}
        result = dataset._format_output(item, attrs)
        assert isinstance(result, dict)
        assert 'image' in result
        assert 'metadata' in result
        assert torch.equal(result['image'], item)
        assert result['metadata'] == attrs

    def test_context_manager(self, create_test_files):
        """Test context manager functionality."""
        with DummyFileDataset(path=str(create_test_files[0])) as dataset:
            assert len(dataset) > 0
            item = dataset[0]
            assert isinstance(item, torch.Tensor)
        # After exiting, should be closed
        assert len(dataset._mmap) == 0

    def test_close_method(self, create_test_files):
        """Test close method."""
        dataset = DummyFileDataset(path=str(create_test_files[0]))
        assert len(dataset._mmap) > 0
        dataset.close()
        assert len(dataset._mmap) == 0
        assert len(dataset._file_offsets) == 0

    def test_empty_dataset(self):
        """Test dataset with no files."""
        dataset = DummyFileDataset(path=None)
        assert len(dataset) == 0
        assert dataset.shape == ()
        assert dataset.n_samples == 0

    def test_empty_file_list(self, temp_dir):
        """Test dataset with empty file list."""
        dataset = DummyFileDataset(path=[])
        assert len(dataset) == 0

    def test_nonexistent_path(self, temp_dir):
        """Test dataset with nonexistent path."""
        pattern = str(temp_dir / "nonexistent" / "*.npy")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dataset = DummyFileDataset(path=pattern)
            assert dataset.n_files == 0

    def test_mixed_file_types(self, temp_dir):
        """Test handling of mixed file types."""
        # Create files with different extensions
        np.save(temp_dir / "file1.npy", np.random.randn(5, 3))
        (temp_dir / "file2.txt").write_text("test")    
        pattern = str(temp_dir / "*")
        dataset = DummyFileDataset(path=pattern)
        # Should only load .npy files
        assert all(f.suffix in [".npy"] for f in dataset.files)

    def test_single_sample_dataset(self, temp_dir):
        """Test dataset with single sample."""
        data = np.random.randn(1, 5, 5).astype(np.float32)
        file_path = temp_dir / "single.npy"
        np.save(file_path, data)
        dataset = DummyFileDataset(path=str(file_path))
        assert len(dataset) == 1
        assert dataset[0].shape == (5, 5)

    def test_different_dtypes(self, temp_dir):
        """Test dataset with different dtype."""
        data = np.random.randn(10, 5, 5).astype(np.float64)
        file_path = temp_dir / "float64.npy"
        np.save(file_path, data)
        dataset = DummyFileDataset(
            path=str(file_path),
            dtype=torch.float64
        )
        item = dataset[0]
        assert item.dtype == torch.float64

    def test_pathlib_path_input(self, create_test_files):
        """Test initialization with pathlib.Path object."""
        dataset = DummyFileDataset(path=create_test_files[0])
        assert dataset.n_files == 1

    def test_list_of_pathlib_paths(self, create_test_files):
        """Test initialization with list of Path objects."""
        dataset = DummyFileDataset(path=create_test_files)
        assert dataset.n_files == 3

    def test_rebuild_index_after_modification(self, temp_dir, create_test_files):
        """Test that index is rebuilt correctly."""
        dataset = DummyFileDataset(path=create_test_files)
        original_offsets = dataset._file_offsets.copy()
        # Force rebuild
        dataset._file_offsets = []
        dataset._build_index()
        assert dataset._file_offsets == original_offsets

    def test_full_workflow_file_dataset(self, temp_dir, create_test_files):
        """Test complete workflow with FileDataset."""
        with DummyFileDataset(path=create_test_files) as dataset:
            # Check basic properties
            assert dataset.n_files == 3
            assert len(dataset) == 33
            # Access various indices
            first = dataset[0]
            middle = dataset[16]
            last = dataset[-1]
            assert all(isinstance(item, torch.Tensor) for item in [first, middle, last])
            assert all(item.shape == (5, 5) for item in [first, middle, last])

    def test_multifile_indexing(self, temp_dir, sample_data):
        """Test correct indexing across multiple files."""
        # Create files with known data
        files = []
        for i, (name, data) in enumerate(sample_data.items()):
            file_path = temp_dir / f"file{i}.npy"
            np.save(file_path, data)
            files.append(file_path)
        dataset = DummyFileDataset(path=[str(f) for f in files])
        # Verify we can access all samples
        for i in range(len(dataset)):
            item = dataset[i]
            assert item.shape == (5, 5)

    def test_large_number_of_files(self, temp_dir):
        """Test handling many files."""
        n_files = 50
        files = []
        for i in range(n_files):
            data = np.random.randn(5, 3, 3).astype(np.float32)
            file_path = temp_dir / f"file{i:03d}.npy"
            np.save(file_path, data)
            files.append(file_path)
        pattern = str(temp_dir / "*.npy")
        dataset = DummyFileDataset(path=pattern)
        assert dataset.n_files == n_files
        assert len(dataset) == n_files * 5

    def test_deep_nesting(self, temp_dir):
        """Test handling deeply nested directory structure."""
        # Create deep nesting
        deep_path = temp_dir
        for i in range(10):
            deep_path = deep_path / f"level{i}"
            deep_path.mkdir(exist_ok=True)
        # Create file at deepest level
        data = np.random.randn(5, 3, 3).astype(np.float32)
        file_path = deep_path / "deep_file.npy"
        np.save(file_path, data)
        # Should find it with recursive glob
        pattern = str(temp_dir / "**" / "*.npy")
        dataset = DummyFileDataset(path=pattern)
        assert dataset.n_files == 1


class TestCachingDataset:
    """Test CachingDataset initialization."""

    def test_init_default_cache(self, create_test_files):
        """Test initialization with default cache size."""
        dataset = DummyCachingDataset(path=str(create_test_files[0]))
        assert dataset.cache is not None
        assert dataset.cache_size > nbytes(0)
        print(dataset.cache)
        dataset.close()

    def test_init_custom_cache_size(self, create_test_files):
        """Test initialization with custom cache size."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="1M"
        )
        assert dataset.cache_size == nbytes("1M")
        dataset.close()

    def test_init_no_cache(self, create_test_files):
        """Test initialization with cache disabled."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache=None
        )
        # Cache should still exist but with size 0
        assert dataset.cache is not None
        dataset.close()

    def test_init_with_attrs_cache(self, create_test_files):
        """Test initialization with attributes cache."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            attrs_cache="1M"
        )
        assert dataset.attrs_cache is not None
        dataset.close()

    def test_init_with_preload(self, create_test_files):
        """Test initialization with preload option."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M",
            preload=True
        )
        assert dataset.n_cached > 0
        dataset.close()

    def test_n_cached_property(self, create_test_files):
        """Test n_cached property."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        # Initially should be 0
        initial_cached = dataset.n_cached
        # Access an item to cache it
        _ = dataset[0]
        assert initial_cached == 0
        assert dataset.n_cached > initial_cached
        dataset.close()

    def test_n_cacheable_property(self, create_test_files):
        """Test n_cacheable property."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        assert dataset.n_cacheable >= 0
        dataset.close()

    def test_cached_bytes_property(self, create_test_files):
        """Test cached_bytes property."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        assert isinstance(dataset.cached_bytes, nbytes)
        _ = dataset[0]
        assert dataset.cached_bytes > 0
        dataset.close()

    def test_cached_bytes_total_property(self, create_test_files):
        """Test cached_bytes_total property."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M",
            attrs_cache="1M"
        )
        assert isinstance(dataset.cached_bytes_total, nbytes)
        _ = dataset[0]
        assert dataset.cached_bytes_total > 0
        dataset.close()

    def test_cache_size_property(self, create_test_files):
        """Test cache_size property."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="5M"
        )
        assert dataset.cache_size == nbytes("5M")
        dataset.close()

    def test_cache_size_total_property(self, create_test_files):
        """Test cache_size_total property."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="5M",
            attrs_cache="1M"
        )
        expected = nbytes("5M") + nbytes("1M")
        assert dataset.cache_size_total == expected
        dataset.close()

    def test_sample_size_property(self, create_test_files):
        """Test sample_size property."""
        dataset = DummyCachingDataset(path=str(create_test_files[0]))
        # Sample shape is (5, 5), dtype is float32 (4 bytes)
        expected = nbytes(5 * 5 * 4)
        assert dataset.sample_size == expected
        dataset.close()

    def test_serial_size_property(self, create_test_files):
        """Test serial_size property."""
        dataset = DummyCachingDataset(path=str(create_test_files[0]))
        # 10 samples of (5, 5) float32
        expected = nbytes(10 * 5 * 5 * 4)
        assert dataset.serial_size == expected
        dataset.close()

    def test_cache_item(self, create_test_files):
        """Test caching an item."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        item = torch.randn(5, 5)
        dataset.cache_item(0, item)
        cached = dataset.get_cached(0)
        if cached is not None:
            assert torch.equal(cached, item)
        dataset.close()

    def test_get_cached_miss(self, create_test_files):
        """Test getting uncached item returns None."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        cached = dataset.get_cached(5)
        assert cached is None
        dataset.close()

    def test_cache_item_overwrite(self, create_test_files):
        """Test overwriting cached item."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        item1 = torch.randn(5, 5)
        item2 = torch.randn(5, 5)
        dataset.cache_item(0, item1)
        dataset.cache_item(0, item2, overwrite=True)
        cached = dataset.get_cached(0)
        if cached is not None:
            assert torch.equal(cached, item2)
        dataset.close()

    def test_cache_attrs(self, create_test_files):
        """Test caching attributes."""
        dataset = DummyCachingDataset(
            path=create_test_files,
            cache="10M",
            attrs_cache="1M",
            has_attrs=True,
        )
        attrs = {'id': 0, 'label': 'test'}
        dataset.cache_attrs(0, attrs)
        cached_attrs = dataset.get_cached_attrs(0)
        assert cached_attrs == attrs
        dataset.close()

    def test_get_cached_attrs_miss(self, create_test_files):
        """Test getting uncached attrs returns None."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M",
            attrs_cache="1M"
        )
        cached = dataset.get_cached_attrs(5)
        assert cached is None

    def test_clear_cache(self, create_test_files):
        """Test clearing the cache."""
        dataset = DummyCachingDataset(
            path=create_test_files,
            cache="10M"
        )
        dataset.cache_item(0, torch.randn(5, 5))
        dataset.cache_item(1, torch.randn(5, 5))
        dataset.purge_cache()
        assert dataset.get_cached(0) is None
        assert dataset.get_cached(1) is None

    def test_purge_cache(self, create_test_files):
        """Test purging the cache."""
        dataset = DummyCachingDataset(
            path=create_test_files,
            cache="10M"
        )
        original_cache_size = dataset.cache_size
        dataset.purge_cache(reset=True)
        assert dataset.cache_size == original_cache_size

    def test_purge_cache_no_reset(self, create_test_files):
        """Test purging cache without reset."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        dataset.purge_cache(reset=False)
        assert dataset.cache_size == nbytes(0)

    def test_getitem_with_cache(self, create_test_files):
        """Test __getitem__ uses cache."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        )
        item1 = dataset[0]
        item2 = dataset[0]
        assert torch.equal(item1, item2)

    def test_context_manager_purges_cache(self, create_test_files):
        """Test that context manager purges cache on exit."""
        with DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="10M"
        ) as dataset:
            _ = dataset[0]
            initial_cache_size = dataset.cache_size
            assert initial_cache_size == nbytes("10M")
        assert dataset.cache_size == nbytes(0)

    def test_very_large_cache_size(self, create_test_files):
        """Test with cache size larger than dataset."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="1G"  # much larger than needed
        )
        assert dataset.cache is not None

    def test_very_small_cache_size(self, create_test_files):
        """Test with very small cache size."""
        dataset = DummyCachingDataset(
            path=str(create_test_files[0]),
            cache="1K"  # very small
        )
        assert dataset.cache is not None

    def test_full_workflow_caching_dataset(self, create_test_files):
        """Test complete workflow with CachingDataset."""
        with DummyCachingDataset(
            path=create_test_files,
            cache="10M",
            attrs_cache="1M",
            has_attrs=False,
        ) as dataset:
            # Access items multiple times
            for i in [0, 5, 10, 15, 20]:
                item1 = dataset[i]
                item2 = dataset.get_cached(i)
                assert torch.all(item1 == item2)

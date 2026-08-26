# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for ZipDataset."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from chuchichaestli.data.zip import ZipDataset
from chuchichaestli.data.base import CachingDataset
from chuchichaestli.data.cache import nbytes


class DummyDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, data, return_attrs: bool = False):
        """Constructor."""
        self.data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        self.return_attrs = return_attrs

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Item getter."""
        item = self.data[idx]
        if self.return_attrs:
            return item, {"index": idx}
        return item

    def close(self):
        """Dummy close the dataset."""
        pass


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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def create_multiple_datasets(temp_dir):
    """Create multiple .npy files for different datasets."""
    datasets_config = {
        "inputs": np.random.randn(10, 5, 5).astype(np.float32),
        "targets": np.random.randn(10, 3, 3).astype(np.float32),
        "masks": np.random.randint(0, 2, (10, 5, 5)).astype(np.int64),
    }
    paths = {}
    for name, data in datasets_config.items():
        dir_path = temp_dir / name
        dir_path.mkdir()
        file_path = dir_path / f"{name}.npy"
        np.save(file_path, data)
        paths[name] = file_path
    return paths


@pytest.fixture
def create_different_size_datasets(temp_dir):
    """Create datasets with different sizes."""
    configs = [
        ("ds1", 15, (5, 5)),
        ("ds2", 10, (3, 3)),
        ("ds3", 20, (7, 7)),
    ]
    paths = []
    for name, n_samples, sample_shape in configs:
        data = np.random.randn(n_samples, *sample_shape).astype(np.float32)
        file_path = temp_dir / f"{name}.npy"
        np.save(file_path, data)
        paths.append(file_path)
    return paths


class TestZipDataset:
    """Tests for ZipDataset."""

    def test_init_single_dataset(self):
        """Test initialization with single dataset."""
        dataset = DummyDataset([1, 2, 3])
        zipped = ZipDataset(dataset)
        assert zipped.n_datasets == 1
        assert len(zipped) == 3

    def test_init_multiple_datasets(self):
        """Test initialization with multiple datasets."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        dataset3 = DummyDataset([7, 8, 9])
        zipped = ZipDataset(dataset1, dataset2, dataset3)
        assert zipped.n_datasets == 3
        assert len(zipped) == 3

    def test_init_no_datasets_raises(self):
        """Test that initialization with no datasets raises ValueError."""
        with pytest.raises(ValueError, match="At least one dataset"):
            ZipDataset()

    def test_length_minimum(self):
        """Test that length is determined by shortest dataset."""
        dataset1 = DummyDataset([1, 2, 3, 4, 5])
        dataset2 = DummyDataset([1, 2, 3])
        dataset3 = DummyDataset([1, 2, 3, 4])
        zipped = ZipDataset(dataset1, dataset2, dataset3)
        assert len(zipped) == 3

    def test_strict_mode_same_length(self):
        """Test strict mode with same-length datasets."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2, strict=True)
        assert len(zipped) == 3

    def test_strict_mode_different_length_raises(self):
        """Test strict mode with different-length datasets raises."""
        dataset1 = DummyDataset([1, 2, 3, 4])
        dataset2 = DummyDataset([1, 2, 3])
        with pytest.raises(ValueError, match="same length when `strict=True`"):
            ZipDataset(dataset1, dataset2, strict=True)

    def test_getitem_tuple_return(self):
        """Test __getitem__ with tuple return format."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2, zip_as="tuple")
        item = zipped[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert item[0] == 1
        assert item[1] == 4

    def test_getitem_dict_return(self):
        """Test __getitem__ with dict return format."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2, zip_as="dict")
        item = zipped[0]
        assert isinstance(item, dict)
        assert len(item) == 2
        assert item[0] == 1
        assert item[1] == 4

    def test_getitem_custom_dict_return(self):
        """Test __getitem__ with custom dict mapping."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        dataset3 = DummyDataset([7, 8, 9])
        zipped = ZipDataset(
            dataset1, dataset2, dataset3, zip_as={"input": 0, "target": 1, "mask": 2}
        )
        item = zipped[0]
        assert isinstance(item, dict)
        assert item["input"] == 1
        assert item["target"] == 4
        assert item["mask"] == 7

    def test_getitem_custom_dict_missing_index(self):
        """Test custom dict with out-of-range index."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        with pytest.warns(UserWarning, match="out of range"):
            zipped = ZipDataset(dataset1, dataset2, zip_as={"input": 0, "invalid": 5})
            item = zipped[0]
            assert "input" in item
            assert "invalid" not in item

    def test_getitem_negative_index(self):
        """Test __getitem__ with negative index."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2)
        item = zipped[-1]
        assert item == (3, 6)

    def test_getitem_out_of_range_positive(self):
        """Test __getitem__ raises IndexError for positive out of range."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2)
        with pytest.raises(IndexError, match="out of range"):
            _ = zipped[10]

    def test_getitem_out_of_range_negative(self):
        """Test __getitem__ raises IndexError for negative out of range."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2)
        with pytest.raises(IndexError, match="out of range"):
            _ = zipped[-10]

    def test_iteration(self):
        """Test iterating over ZipDataset."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2)
        first = zipped[0]
        items = list(zipped)
        assert len(items) == 3
        assert len(first) == 2
        assert items[1] == (2, 5)
        assert items[2] == (3, 6)

    def test_str_representation(self):
        """Test string representation."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        zipped = ZipDataset(dataset1, dataset2)
        s = str(zipped)
        r = repr(zipped)
        assert "ZipDataset" in s
        assert "n=2" in s
        assert "len=3" in s
        assert s == r

    def test_context_manager(self):
        """Test context manager functionality."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        with ZipDataset(dataset1, dataset2) as zipped:
            item = zipped[0]
            assert item == (1, 4)

    def test_close_method(self):
        """Test close method calls close on all datasets."""
        dataset1 = DummyDataset([1, 2, 3])
        dataset2 = DummyDataset([4, 5, 6])
        # Add tracking for close calls
        close_called = {"d1": False, "d2": False}
        original_close1 = dataset1.close
        original_close2 = dataset2.close

        def track_close1():
            close_called["d1"] = True
            original_close1()

        def track_close2():
            close_called["d2"] = True
            original_close2()

        dataset1.close = track_close1
        dataset2.close = track_close2
        zipped = ZipDataset(dataset1, dataset2)
        zipped.close()
        assert close_called["d1"]
        assert close_called["d2"]

    def test_cache_size_total(self, create_multiple_datasets):
        """Test cache_size_total property."""
        paths = create_multiple_datasets
        dataset1 = DummyCachingDataset(paths["inputs"], cache="1M")
        dataset2 = DummyCachingDataset(paths["targets"], cache="2M")
        zipped = ZipDataset(dataset1, dataset2)
        total = zipped.cache_size_total
        assert dataset1.cache_size == nbytes("1M")
        assert dataset2.cache_size == nbytes("2M")
        assert total == nbytes("3M")

    def test_purge_cache(self, create_multiple_datasets):
        """Test purge_cache method."""
        paths = create_multiple_datasets
        dataset1 = DummyCachingDataset(paths["inputs"], cache="1M")
        dataset2 = DummyCachingDataset(paths["targets"], cache="2M")
        # Simulate some cached data
        zipped = ZipDataset(dataset1, dataset2)
        assert zipped.cache_size_total > 0
        zipped.purge_cache(reset=True)
        zipped.close()
        assert zipped.cache_size_total == nbytes(0)

    def test_empty_dataset_warning(self):
        """Test warning when one dataset is empty."""
        dataset1 = DummyDataset([])
        dataset2 = DummyDataset([1, 2, 3])
        with pytest.warns(UserWarning, match="length 0"):
            zipped = ZipDataset(dataset1, dataset2)
            assert len(zipped) == 0

    def test_with_tensor_data(self):
        """Test with actual tensor data."""
        data1 = torch.randn(10, 5, 5)
        data2 = torch.randn(10, 3, 3)
        dataset1 = DummyDataset(data1)
        dataset2 = DummyDataset(data2)
        zipped = ZipDataset(dataset1, dataset2)
        item1, item2 = zipped[0]
        assert item1.shape == (5, 5)
        assert item2.shape == (3, 3)
        assert torch.equal(item1, data1[0])
        assert torch.equal(item2, data2[0])

    def test_with_attributes(self):
        """Test zipping datasets that return (data, attrs) tuples."""
        dataset1 = DummyDataset([1, 2, 3], return_attrs=True)
        dataset2 = DummyDataset([4, 5, 6], return_attrs=True)
        zipped = ZipDataset(dataset1, dataset2)
        (item1, attrs1), (item2, attrs2) = zipped[0]
        assert item1 == 1
        assert item2 == 4
        assert attrs1 == {"index": 0}
        assert attrs2 == {"index": 0}

    def test_many_datasets(self):
        """Test with many datasets."""
        datasets = [DummyDataset(list(range(i, i + 5))) for i in range(10)]
        zipped = ZipDataset(*datasets)
        assert zipped.n_datasets == 10
        item = zipped[0]
        assert len(item) == 10
        assert item[0] == 0
        assert item[5] == 5

    def test_from_paths_basic(self, create_multiple_datasets):
        """Test basic usage of from_paths."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            cache="1M",
        )
        assert zipped.n_datasets == 2
        assert len(zipped) == 10
        zipped.close()

    def test_from_paths_with_preload(self, create_multiple_datasets):
        """Test from_paths with preload option."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            cache="10M",
            preload=True,
        )
        # Check that data is cached
        assert zipped.datasets[0].n_cached > 0
        assert zipped.datasets[1].n_cached > 0
        zipped.close()

    def test_from_paths_custom_dtype(self, create_multiple_datasets):
        """Test from_paths with custom dtype."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            dtype=torch.float64,
            cache="1M",
        )
        item1, item2 = zipped[0]
        assert item1.dtype == torch.float64
        assert item2.dtype == torch.float64
        zipped.close()

    def test_from_paths_with_attrs_cache(self, create_multiple_datasets):
        """Test from_paths with attributes cache."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            cache="1M",
            attrs_cache="500K",
        )
        assert zipped.datasets[0].attrs_cache is not None
        assert zipped.datasets[1].attrs_cache is not None
        zipped.close()

    def test_from_paths_strict_mode(self, create_different_size_datasets):
        """Test from_paths with strict mode raises on different sizes."""
        paths = create_different_size_datasets
        with pytest.raises(ValueError, match="same length when `strict=True`"):
            ZipDataset.from_paths(
                paths[0],
                paths[1],
                paths[2],
                dataset_cls=DummyCachingDataset,
                strict=True,
                cache="1M",
            )

    def test_from_paths_no_strict_different_sizes(self, create_different_size_datasets):
        """Test from_paths without strict mode accepts different sizes."""
        paths = create_different_size_datasets
        zipped = ZipDataset.from_paths(
            paths[0],  # 15 samples
            paths[1],  # 10 samples
            paths[2],  # 20 samples
            dataset_cls=DummyCachingDataset,
            strict=False,
            cache="1M",
        )
        # Length should be minimum (10)
        assert len(zipped) == 10
        zipped.close()

    def test_from_paths_no_paths_raises(self):
        """Test from_paths with no paths raises ValueError."""
        with pytest.raises(ValueError, match="At least one path"):
            ZipDataset.from_paths(dataset_cls=DummyCachingDataset)

    def test_from_paths_wildcard_pattern(self, temp_dir):
        """Test from_paths with wildcard patterns."""
        # Create multiple files in directories
        for i in range(3):
            dir1 = temp_dir / f"input_{i}"
            dir2 = temp_dir / f"target_{i}"
            dir1.mkdir()
            dir2.mkdir()
            np.save(dir1 / f"data_{i}.npy", np.random.randn(5, 3, 3).astype(np.float32))
            np.save(dir2 / f"data_{i}.npy", np.random.randn(5, 3, 3).astype(np.float32))
        zipped = ZipDataset.from_paths(
            str(temp_dir / "input_*" / "*.npy"),
            str(temp_dir / "target_*" / "*.npy"),
            dataset_cls=DummyCachingDataset,
            cache="1M",
        )
        assert zipped.n_datasets == 2
        zipped.close()

    def test_from_paths_return_as_dict(self, create_multiple_datasets):
        """Test from_paths with dict return format."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            zip_as={"input": 0, "target": 1},
            cache="1M",
        )
        sample = zipped[0]
        assert isinstance(sample, dict)
        assert "input" in sample
        assert "target" in sample
        zipped.close()

    def test_from_paths_return_as_invalid_dict(self, create_multiple_datasets):
        """Test from_paths with dict return format."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            zip_as={"input": "invalid", "target": 1},
            cache="1M",
        )
        with pytest.warns(UserWarning, match="Invalid dataset index type for key"):
            _ = zipped[0]
        zipped.close()

    def test_from_named_paths_basic(self, create_multiple_datasets):
        """Test basic usage of from_named_paths."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_named_paths(
            {
                "input": paths["inputs"],
                "target": paths["targets"],
                "mask": paths["masks"],
            },
            dataset_cls=DummyCachingDataset,
            cache="1M",
        )
        assert zipped.n_datasets == 3
        assert len(zipped) == 10
        # Check that return format is dict with correct keys
        sample = zipped[0]
        assert isinstance(sample, dict)
        assert "input" in sample
        assert "target" in sample
        assert "mask" in sample
        zipped.close()

    def test_from_named_paths_preserves_order(self, create_multiple_datasets):
        """Test that from_named_paths preserves key order."""
        paths = create_multiple_datasets
        ordered_paths = {
            "first": paths["inputs"],
            "second": paths["targets"],
            "third": paths["masks"],
        }
        zipped = ZipDataset.from_named_paths(
            ordered_paths, dataset_cls=DummyCachingDataset, cache="1M"
        )
        sample = zipped[0]
        keys = list(sample.keys())
        assert keys == ["first", "second", "third"]
        zipped.close()

    def test_from_named_paths_with_preload(self, create_multiple_datasets):
        """Test from_named_paths with preload."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_named_paths(
            {"input": paths["inputs"], "target": paths["targets"]},
            dataset_cls=DummyCachingDataset,
            cache="10M",
            preload=True,
        )
        assert all(ds.n_cached > 0 for ds in zipped.datasets)
        zipped.close()

    def test_from_named_paths_empty_raises(self):
        """Test from_named_paths with empty dict raises."""
        with pytest.raises(ValueError, match="At least one named path"):
            ZipDataset.from_named_paths({}, dataset_cls=DummyCachingDataset)

    def test_from_named_paths_single_entry(self, create_multiple_datasets):
        """Test from_named_paths with single entry."""
        paths = create_multiple_datasets
        zipped = ZipDataset.from_named_paths(
            {"data": paths["inputs"]}, dataset_cls=DummyCachingDataset, cache="1M"
        )
        assert zipped.n_datasets == 1
        sample = zipped[0]
        assert isinstance(sample, dict)
        assert "data" in sample
        zipped.close()

    def test_full_workflow_from_paths(self, create_multiple_datasets):
        """Test complete workflow using from_paths."""
        paths = create_multiple_datasets
        with ZipDataset.from_paths(
            paths["inputs"],
            paths["targets"],
            dataset_cls=DummyCachingDataset,
            zip_as={"input": 0, "target": 1},
            cache="10M",
            preload=True,
            strict=True,
        ) as zipped:
            # Access data
            for i in range(min(5, len(zipped))):
                sample = zipped[i]
                assert "input" in sample
                assert "target" in sample
                assert sample["input"].shape == (5, 5)
                assert sample["target"].shape == (3, 3)
            # Check caching
            assert zipped.cached_bytes_total > nbytes(0)

    def test_full_workflow_from_named_paths(self, create_multiple_datasets):
        """Test complete workflow using from_named_paths."""
        paths = create_multiple_datasets
        with ZipDataset.from_named_paths(
            {
                "image": paths["inputs"],
                "label": paths["targets"],
                "attention": paths["masks"],
            },
            dataset_cls=DummyCachingDataset,
            cache="5M",
            preload=True,
        ) as zipped:
            # Verify structure
            sample = zipped[0]
            assert set(sample.keys()) == {"image", "label", "attention"}
            # Verify iteration
            for sample in zipped:
                assert "image" in sample
                assert "label" in sample
                assert "attention" in sample


class _Offsets(DummyDataset):
    """A dataset advertising file structure, like any `FileDataset`."""

    def __init__(self, data, offsets, files=None):
        """Constructor."""
        super().__init__(data)
        self._file_offsets = list(offsets)
        self.files = [Path(f) for f in (files or [])]


class TestZipDatasetIndexStructure:
    """Forwarding of sequence boundaries to samplers and collates."""

    def test_offsets_are_forwarded_when_constituents_agree(self):
        """A window sampler must still see where each source sequence ends."""
        a = _Offsets(np.zeros((10, 2)), [0, 6, 10])
        b = _Offsets(np.zeros((10, 3)), [0, 6, 10])
        assert ZipDataset(a, b)._file_offsets == [0, 6, 10]

    def test_disagreeing_offsets_are_unioned(self):
        """A window must not straddle a discontinuity in *any* zipped stream."""
        a = _Offsets(np.zeros((10, 2)), [0, 6, 10])
        b = _Offsets(np.zeros((10, 3)), [0, 4, 10])
        assert ZipDataset(a, b)._file_offsets == [0, 4, 6, 10]

    def test_a_structureless_dataset_contributes_nothing(self):
        """Datasets without file structure are continuous."""
        a = _Offsets(np.zeros((10, 2)), [0, 6, 10])
        zipped = ZipDataset(a, DummyDataset(np.zeros((10, 3))))
        assert zipped._file_offsets == [0, 6, 10]

    def test_absent_offsets_give_none(self):
        """Plain datasets carry no structure to forward."""
        zipped = ZipDataset(
            DummyDataset(np.zeros((8, 2))), DummyDataset(np.zeros((8, 3)))
        )
        assert zipped._file_offsets is None

    def test_offsets_are_clipped_to_the_shortest_dataset(self):
        """`strict=False` truncates the zip, so the boundaries must follow."""
        a = _Offsets(np.zeros((10, 2)), [0, 6, 10])
        b = DummyDataset(np.zeros((7, 3)))
        zipped = ZipDataset(a, b)
        assert len(zipped) == 7
        assert zipped._file_offsets == [0, 6, 7]

    def test_differing_lengths_do_not_warn(self):
        """`strict=False` with different sizes is ordinary use, not a problem."""
        import warnings

        a = _Offsets(np.zeros((10, 2)), [0, 10])
        b = _Offsets(np.zeros((7, 3)), [0, 7])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zipped = ZipDataset(a, b)
        assert not caught
        assert zipped._file_offsets == [0, 7]

    def test_truncation_past_the_first_boundary(self):
        """Clipping must not leave a boundary beyond the end."""
        a = _Offsets(np.zeros((10, 2)), [0, 6, 10])
        zipped = ZipDataset(a, DummyDataset(np.zeros((4, 3))))
        assert zipped._file_offsets == [0, 4]

    def test_files_come_from_the_first_reporting_dataset(self):
        """Provenance needs one set of paths to label a batch."""
        a = _Offsets(np.zeros((10, 2)), [0, 10], files=["a0.npy"])
        b = _Offsets(np.zeros((10, 3)), [0, 10], files=["b0.npy"])
        assert ZipDataset(a, b).files == [Path("a0.npy")]

    def test_files_is_none_without_any(self):
        """No constituent reports paths."""
        assert ZipDataset(DummyDataset(np.zeros((4, 2)))).files is None

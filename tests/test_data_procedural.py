# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for ProceduralDataset and derived classes."""

import torch
import pytest
# from safetensors import safe_open
# from safetensors.torch import save_file as st_save
from chuchichaestli.data.procedural import (
    ProceduralDataset,
    HalfMoonsDataset,
    SpiralsDataset,
    CheckerboardDataset,
    RingsDataset,
    ConcentricSpheresDataset,
    GaussiansDataset,
    SwissRollDataset,
    generate_procedural_dataset,
)


N = 20


DATASET_FACTORIES = [
    pytest.param(lambda: HalfMoonsDataset(n_samples=N, noise=0.0, seed=0), id="HalfMoons"),
    pytest.param(lambda: SpiralsDataset(n_samples=N, noise=0.0, seed=0), id="Spirals"),
    pytest.param(lambda: CheckerboardDataset(n_samples=N, noise=0.0, seed=0), id="Checkerboard"),
    pytest.param(lambda: RingsDataset(n_samples=N, n_rings=3, noise=0.0, seed=0), id="Rings"),
    pytest.param(lambda: ConcentricSpheresDataset(dim=3, n_samples=N, noise=0.0, seed=0), id="ConcentricSpheres"),
    pytest.param(lambda: GaussiansDataset(dim=2, n_samples=N // 3, n_gaussians=3, noise=0.0, seed=0), id="Gaussians"),
    pytest.param(lambda: SwissRollDataset(n_samples=N, noise=0.0, seed=0), id="SwissRoll"),
]


def _xy(ds: ProceduralDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the full (X, y) tensors from a dataset's _mmap."""
    data = ds._mmap[0]
    return data[:, : ds.dim], data[:, ds.dim]


class TestProceduralDatasetAbstract:
    """Test the ProceduralDataset class."""

    def test_cannot_instantiate_directly(self):
        """Test abstract instance."""
        with pytest.raises(TypeError):
            ProceduralDataset(dim=2, n_samples=10)

    def test_subclass_without_generate_raises(self):
        """Test incomplete abstract instance."""
        class Incomplete(ProceduralDataset):
            pass  # generate() not implemented
        with pytest.raises(TypeError):
            Incomplete(dim=2, n_samples=10)

    def test_repr_format(self):
        """Test representation format."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.0, seed=0)
        r = repr(ds)
        print(r)
        assert "HalfMoonsDataset" in r
        assert str(N) in r
        assert "2" in r  # default dim


class TestDerivedProceduralDataset:
    """Tests run against every pre-defined ProceduralDataset class."""

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_len_equals_n_samples(self, factory):
        """Test n_samples."""
        ds = factory()
        assert len(ds) == ds.n_samples

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_x_shape(self, factory):
        """Test x shape."""
        ds = factory()
        X, _ = _xy(ds)
        assert X.shape == (ds.n_samples, ds.dim)

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_y_shape(self, factory):
        """Test y shape."""
        ds = factory()
        _, y = _xy(ds)
        assert y.shape == (ds.n_samples,)

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_dtype_float32(self, factory):
        """Test data type."""
        ds = factory()
        X, y = _xy(ds)
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_getitem_returns_feature_label_pair(self, factory):
        """Test item getter."""
        ds = factory()
        x_i, y_i = ds[0]
        assert x_i.shape == (ds.dim,)
        assert y_i.shape == ()

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_getitem_negative_index(self, factory):
        """Test negative item index."""
        ds = factory()
        x_last, y_last = ds[-1]
        x_exp, y_exp = ds[len(ds) - 1]
        assert torch.equal(x_last, x_exp)
        assert torch.equal(y_last, y_exp)

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_getitem_out_of_bounds(self, factory):
        """Test out-of-bounds index."""
        ds = factory()
        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_reproducibility_same_seed(self, factory):
        """Test random seed."""
        ds_a = factory()
        ds_b = factory()
        X_a, y_a = _xy(ds_a)
        X_b, y_b = _xy(ds_b)
        assert torch.equal(X_a, X_b)
        assert torch.equal(y_a, y_b)

    def test_different_seeds_differ(self):
        """Test random seeds variability."""
        # Use procedural_dataset with a trivially different generator to check seeds vary.
        def _uniform(ds_):
            torch.manual_seed(ds_.seed)
            X = torch.rand(ds_.n_samples, ds_.dim)
            y = torch.zeros(ds_.n_samples)
            return X, y

        ds_seed_a = generate_procedural_dataset(_uniform, dim=2, n_samples=N, seed=1)
        ds_seed_b = generate_procedural_dataset(_uniform, dim=2, n_samples=N, seed=2)
        X_a, _ = _xy(ds_seed_a)
        X_b, _ = _xy(ds_seed_b)
        assert not torch.equal(X_a, X_b)

    @pytest.mark.parametrize("factory", DATASET_FACTORIES)
    def test_mmap_populated(self, factory):
        """Test _mmap content."""
        ds = factory()
        assert len(ds._mmap) == 1
        assert ds._mmap[0].shape[1] == ds.dim + 1

    @pytest.mark.parametrize("n_samples", [7, 10, 11, 13, 17, 99, 100, 101])
    @pytest.mark.parametrize("cls", [RingsDataset, HalfMoonsDataset, SpiralsDataset, ConcentricSpheresDataset])
    def test_classes_exact_n_samples(self, cls, n_samples):
        """Test derived ProceduralDataset n_samples per class."""
        ds = cls(n_samples=n_samples, seed=0)
        assert len(ds) == n_samples

    @pytest.mark.parametrize("n_rings", [2, 3, 4, 5])
    def test_rings_exact_for_varying_n_rings(self, n_rings):
        """Test RingsDataset's n_samples with varying number of rings."""
        n_samples = 17  # not divisible by 2, 3, 4, or 5
        ds = RingsDataset(n_samples=n_samples, n_rings=n_rings, seed=0)
        assert len(ds) == n_samples

    def test_halfmoons_binary_labels(self):
        """Test HalfMoons classes."""
        ds = HalfMoonsDataset(n_samples=N, seed=0)
        _, y = _xy(ds)
        assert set(y.long().tolist()).issubset({0, 1})

    def test_spirals_binary_labels(self):
        """Test Spirals classes."""
        ds = SpiralsDataset(n_samples=N, seed=0)
        _, y = _xy(ds)
        assert set(y.long().tolist()).issubset({0, 1})

    def test_checkerboard_binary_labels(self):
        """Test Checkerboard classes."""
        ds = CheckerboardDataset(n_samples=N, seed=0)
        _, y = _xy(ds)
        assert set(y.long().tolist()).issubset({0, 1})

    def test_rings_label_range(self):
        """Test Rings classes."""
        n_rings = 4
        ds = RingsDataset(n_samples=N, n_rings=n_rings, seed=0)
        _, y = _xy(ds)
        labels = set(y.long().tolist())
        assert labels == set(range(n_rings))

    def test_concentric_spheres_binary_labels(self):
        """Test ConcentricSpheres classes."""
        ds = ConcentricSpheresDataset(n_samples=N, seed=0)
        _, y = _xy(ds)
        assert set(y.long().tolist()).issubset({0, 1})

    def test_gaussians_label_range(self):
        """Test Gaussians classes."""
        n_gaussians = 5
        ds = GaussiansDataset(n_samples=N, n_gaussians=n_gaussians, seed=0)
        _, y = _xy(ds)
        labels = set(y.long().tolist())
        assert labels == set(range(n_gaussians))

    def test_swissroll_continuous_label_in_unit_interval(self):
        """SwissRoll classes."""
        ds = SwissRollDataset(n_samples=N, noise=0.0, seed=0)
        _, y = _xy(ds)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    @pytest.mark.parametrize("dim", [2, 3, 5, 8])
    @pytest.mark.parametrize("cls", [RingsDataset, HalfMoonsDataset, SpiralsDataset, ConcentricSpheresDataset])
    def test_arbitrary_dim(self, cls, dim):
        """Test derived ProceduralDatasets in various dimensions."""
        ds = cls(n_samples=N, noise=0.05, dim=dim, seed=0)
        assert ds.dim == dim
        X, _ = _xy(ds)
        assert X.shape == (N, dim)

    def test_checkerboard_2d_matches_parity(self):
        """In 2D the label is still (ix + iy) % 2."""
        ds = CheckerboardDataset(n_samples=200, n_tiles=4, extent=2.0, noise=0.0, dim=2, seed=0)
        X, y = _xy(ds)
        e, n_tiles = ds.extent, ds.n_tiles
        cell_size = 2.0 * e / n_tiles
        ix = torch.floor((X[:, 0] + e) / cell_size).long()
        iy = torch.floor((X[:, 1] + e) / cell_size).long()
        expected_y = ((ix + iy) % 2).float()
        assert torch.equal(y, expected_y)

    def test_regenerate_same_seed_reproduces_data(self):
        """Test ProceduralDataset.regenerate."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.1, seed=7)
        X_orig, y_orig = _xy(ds)
        ds.regenerate()  # seed unchanged
        X_new, y_new = _xy(ds)
        assert torch.equal(X_orig, X_new)
        assert torch.equal(y_orig, y_new)

    def test_regenerate_new_seed_changes_data(self):
        """Test ProceduralDataset.regenerate with different seed."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.1, seed=7)
        X_orig, _ = _xy(ds)
        ds.regenerate(seed=99)
        X_new, _ = _xy(ds)
        assert not torch.equal(X_orig, X_new)

    def test_regenerate_preserves_n_samples_and_dim(self):
        """Test ProceduralDataset.regenerate effect on n_samples/dims."""
        ds = RingsDataset(n_samples=N, n_rings=3, seed=1)
        ds.regenerate(seed=2)
        assert len(ds) == N
        X, _ = _xy(ds)
        assert X.shape == (N, 2)

    def test_save_appends_extension(self, tmp_path):
        """Test ProceduralDataset.save method (no extension)."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.0, seed=0)
        path = ds.save(tmp_path / "moons")  # no extension
        assert path.suffix == ".safetensors"
        assert path.exists()
        
    def test_save_keeps_existing_extension(self, tmp_path):
        """Test ProceduralDataset.save method (with extension)."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.0, seed=0)
        path = ds.save(tmp_path / "moons.safetensors")
        assert path.suffix == ".safetensors"

    def test_roundtrip_preserves_tensors(self, tmp_path):
        """Test ProceduralDataset with path input."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.0, seed=0)
        X_orig, y_orig = _xy(ds)
        save_path = ds.save(tmp_path / "moons")
        ds2 = HalfMoonsDataset(n_samples=N, noise=0.0, path=save_path, seed=99)
        X_loaded, y_loaded = _xy(ds2)
        assert torch.allclose(X_orig, X_loaded)
        assert torch.allclose(y_orig, y_loaded)

    def test_load_n_samples_mismatch_warns(self, tmp_path):
        """Test ProceduralDataset.load number of samples mismatch warning."""
        ds = HalfMoonsDataset(n_samples=N, noise=0.0, seed=0)
        save_path = ds.save(tmp_path / "moons")
        with pytest.warns(UserWarning, match="samples"):
            ds2 = HalfMoonsDataset(n_samples=N + 10, noise=0.0, path=save_path)
        assert len(ds2) == N  # file wins

    def test_save_raises_when_no_data(self):
        """Test ProceduralDataset.save after data erasure."""
        ds = HalfMoonsDataset(n_samples=N, seed=0)
        ds._mmap = []  # simulate empty state
        with pytest.raises(RuntimeError):
            ds.save("/tmp/should_not_exist.safetensors")

    def test_nonexistent_path_warns_and_generates(self, tmp_path):
        """Test ProceduralDataset with non-existing path."""
        nonexistent = tmp_path / "does_not_exist.safetensors"
        with pytest.warns(UserWarning, match="does not exist"):
            with pytest.warns(UserWarning, match="files not found"):
                ds = HalfMoonsDataset(n_samples=N, path=str(nonexistent))
        assert len(ds) == N

class TestGenerateProceduralDataset:
    """Test the generate_procedural_dataset method."""

    def _simple_gen(self, ds):
        torch.manual_seed(ds.seed)
        X = torch.rand(ds.n_samples, ds.dim)
        y = (X[:, 0] > 0.5).float()
        return X, y
    
    def test_basic_shape(self):
        """Test generated dataset shape."""
        ds = generate_procedural_dataset(self._simple_gen, dim=3, n_samples=N)
        assert len(ds) == N
        X, y = _xy(ds)
        assert X.shape == (N, 3)
        assert y.shape == (N,)

    def test_dtype_float64(self):
        """Test generated dataset data type."""
        ds = generate_procedural_dataset(self._simple_gen, dim=2, n_samples=N, dtype=torch.float64)
        X, y = _xy(ds)
        assert X.dtype == torch.float64
        assert y.dtype == torch.float64

    def test_seed_forwarded(self):
        """Test generated dataset seed."""
        ds = generate_procedural_dataset(self._simple_gen, dim=2, n_samples=N, seed=7)
        assert ds.seed == 7

    def test_lambda_name_fallback(self):
        """Test generated dataset with lambda function."""
        ds = generate_procedural_dataset(
            lambda ds: (torch.randn(ds.n_samples, ds.dim), torch.zeros(ds.n_samples)),
            dim=2,
            n_samples=N,
        )
        # Lambda name is "<lambda>"; just check the instance is valid.
        assert len(ds) == N

    def test_extra_attr_accessible_during_generate(self):
        """The callable must be able to read custom attrs set before generate()."""
        call_log = []

        def recording_gen(ds):
            call_log.append(getattr(ds, "custom_attr", "MISSING"))
            return torch.zeros(ds.n_samples, ds.dim), torch.zeros(ds.n_samples)

        generate_procedural_dataset(recording_gen, dim=2, n_samples=N, custom_attr="hello")
        assert call_log == ["hello"]

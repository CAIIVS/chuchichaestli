# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the N-dimensional data transforms."""

import pytest
import torch
from chuchichaestli.data.transforms import (
    RandomCropND,
    CenterCropND,
    RandomFlipND,
    RandomRot90ND,
    ResizeND,
    PadND,
    Affine,
    HEPTransform,
    LogTransform,
    LogP1Transform,
    MinMaxScale,
    InvMinMaxScale,
    Clamp,
    HounsfieldScale,
    InvHounsfieldScale,
    HounsfieldClamp,
    ZScaleInterval,
    ZScale,
    ChannelExpand,
    ChannelCollapse,
    SequentialTransform,
)


class TestRandomCropND:
    """Tests for RandomCropND."""

    def test_output_shape_and_leading_passthrough(self):
        """Crops the trailing dims, leaving leading (channel) dims intact."""
        vol = torch.rand(2, 8, 8, 8)
        out = RandomCropND((4, 4, 4)).transform(vol, {"starts": (0, 0, 0), "size": (4, 4, 4)})
        assert out.shape == (2, 4, 4, 4)

    def test_same_box_across_leaves(self):
        """Given shared params, two leaves get the identical crop."""
        crop = RandomCropND((4, 4, 4))
        x = torch.arange(6 * 6 * 6).float().reshape(6, 6, 6)
        params = crop.make_params([x])
        assert torch.equal(crop.transform(x, params), crop.transform(x.clone(), params))

    def test_origin_in_bounds(self):
        """Sampled origins never push the crop past an axis."""
        crop = RandomCropND((4, 4, 4))
        x = torch.zeros(5, 5, 5)
        for _ in range(50):
            starts = crop.make_params([x])["starts"]
            assert all(0 <= s <= 1 for s in starts)

    def test_standalone_forward(self):
        """Works as a plain v2 transform on a single tensor."""
        assert RandomCropND((4, 4, 4))(torch.rand(1, 8, 8, 8)).shape == (1, 4, 4, 4)

    def test_int_size_requires_ndim(self):
        """An int size without ndim is rejected."""
        with pytest.raises(ValueError):
            RandomCropND(4)
        assert RandomCropND(4, ndim=3).size == (4, 4, 4)


class TestCenterCropND:
    """Tests for CenterCropND."""

    def test_centered_crop(self):
        """Crops symmetrically about the center."""
        x = torch.arange(6).float().reshape(1, 6)
        out = CenterCropND((4,)).transform(x, {})
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def test_pads_when_target_larger(self):
        """Zero-pads axes whose target exceeds the input."""
        out = CenterCropND((8, 8, 8)).transform(torch.ones(2, 6, 6, 6), {})
        assert out.shape == (2, 8, 8, 8)
        assert out[0, 0, 0, 0] == 0.0  # padded corner


class TestGeometric:
    """Tests for RandomFlipND and RandomRot90ND."""

    def test_flip_shared_across_leaves(self):
        """The same flip is applied to two leaves."""
        f = RandomFlipND(3, p=1.0)
        x = torch.rand(6, 6, 6)
        params = f.make_params([x])
        assert torch.equal(f.transform(x, params), f.transform(x.clone(), params))
        assert params["dims"] == (-3, -2, -1)  # p=1 flips every axis

    def test_flip_p_zero_is_identity(self):
        """p=0 leaves the tensor unchanged."""
        f = RandomFlipND(3, p=0.0)
        x = torch.rand(4, 4, 4)
        assert torch.equal(f.transform(x, f.make_params([x])), x)

    def test_rot90_shape_and_shared(self):
        """Rotation keeps cubic shape and is shared across leaves."""
        r = RandomRot90ND(3)
        x = torch.rand(2, 4, 4, 4)
        params = r.make_params([x])
        out = r.transform(x, params)
        assert out.shape == (2, 4, 4, 4)
        assert torch.equal(out, r.transform(x.clone(), params))

    def test_rot90_requires_2d(self):
        """A single spatial dim cannot form a rotation plane."""
        with pytest.raises(ValueError):
            RandomRot90ND(1)


class TestResizePad:
    """Tests for ResizeND and PadND."""

    def test_resize_bare_and_channel(self):
        """Reaches the target size on bare and channelled inputs."""
        assert ResizeND((3, 3, 3), mode="nearest").transform(torch.rand(6, 6, 6), {}).shape == (3, 3, 3)
        assert ResizeND((3, 3, 3), mode="trilinear").transform(torch.rand(2, 6, 6, 6), {}).shape == (2, 3, 3, 3)

    def test_pad_widths(self):
        """Padding widths are honored per trailing axis."""
        out = PadND(((1, 1), (2, 0), (0, 0))).transform(torch.zeros(2, 6, 6, 6), {})
        assert out.shape == (2, 8, 8, 6)


class TestIntensity:
    """Round-trips and ranges for the value transforms (on bare 2D tensors)."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda x: HEPTransform(8, x.min(), x.max()),
            lambda x: LogTransform(),
            lambda x: LogP1Transform(),
            lambda x: Affine(2.0, 5.0),
            lambda x: MinMaxScale(0.0, 200.0, (-1.0, 1.0)),
        ],
    )
    def test_round_trip_2d(self, make):
        """revert(transform(x)) recovers x on a 2D tensor (no F.normalize dep)."""
        x = torch.rand(8, 8) * 100 + 1.0
        t = make(x)
        assert torch.allclose(t.revert(t.transform(x)), x, atol=1e-3, rtol=1e-3)

    def test_affine_semantics(self):
        """Affine(a, b) computes (x - b) / a."""
        x = torch.arange(4.0)
        assert torch.equal(Affine(2.0, 1.0).transform(x), (x - 1.0) / 2.0)

    def test_logp1_handles_zero(self):
        """LogP1 round-trips data containing zeros."""
        x = torch.rand(8, 8) * 100
        t = LogP1Transform()
        assert torch.allclose(t.revert(t.transform(x)), x, atol=1e-3, rtol=1e-3)

    def test_minmax_hits_feature_range(self):
        """MinMaxScale maps min/max to the feature range bounds."""
        x = torch.linspace(0, 10, 25).reshape(5, 5)
        y = MinMaxScale(feature_range=(-1.0, 1.0)).transform(x)
        assert torch.isclose(y.min(), torch.tensor(-1.0))
        assert torch.isclose(y.max(), torch.tensor(1.0))

    def test_clamp_bounds(self):
        """Clamp respects the given bounds."""
        y = Clamp(0.0, 1.0).transform(torch.tensor([-1.0, 0.5, 2.0]))
        assert y.min() == 0.0 and y.max() == 1.0

    def test_hounsfield_scale(self):
        """HounsfieldScale applies slope*x + intercept and inverts it."""
        x = torch.rand(8, 8) * 4000
        t = HounsfieldScale(slope=1.0, intercept=-1024.0)
        assert torch.equal(t.transform(x), x - 1024.0)
        assert torch.allclose(t.revert(t.transform(x)), x, atol=1e-3)
        inv = t.get_inverse()
        assert isinstance(inv, InvHounsfieldScale)
        assert torch.allclose(inv.transform(t.transform(x)), x, atol=1e-3)
        assert isinstance(inv.get_inverse(), HounsfieldScale)

    def test_hounsfield_clamp_window(self):
        """HounsfieldClamp clips to [center - width/2, center + width/2]."""
        t = HounsfieldClamp(center=40.0, width=400.0)
        assert (t.min, t.max) == (-160.0, 240.0)
        y = t.transform(torch.tensor([-1000.0, 0.0, 3000.0]))
        assert torch.equal(y, torch.tensor([-160.0, 0.0, 240.0]))

    def test_hep_get_inverse(self):
        """get_inverse yields a transform that undoes the forward pass."""
        x = torch.rand(8, 8) * 100 + 1.0
        t = HEPTransform(8, x.min(), x.max())
        inv = t.get_inverse()
        assert torch.allclose(inv.transform(t.transform(x)), x, atol=1e-3, rtol=1e-3)

    def test_minmax_get_inverse(self):
        """MinMaxScale.get_inverse returns an InvMinMaxScale that undoes it."""
        x = torch.rand(8, 8) * 100
        t = MinMaxScale(0.0, 100.0, (-1.0, 1.0))
        inv = t.get_inverse()
        assert isinstance(inv, InvMinMaxScale)
        assert torch.allclose(inv.transform(t.transform(x)), x, atol=1e-3, rtol=1e-3)
        with pytest.raises(RuntimeError):
            MinMaxScale().get_inverse()

    def test_inv_minmax_scale(self):
        """InvMinMaxScale swaps transform/revert and inverts back to MinMaxScale."""
        x = torch.rand(8, 8) * 100
        fwd = MinMaxScale(0.0, 100.0, (-1.0, 1.0))
        inv = InvMinMaxScale(0.0, 100.0, (-1.0, 1.0))
        # inv.transform == fwd.revert (feature -> data), inv.revert == fwd.transform
        assert torch.allclose(inv.transform(fwd.transform(x)), x, atol=1e-3, rtol=1e-3)
        assert torch.allclose(inv.revert(x), fwd.transform(x), atol=1e-3, rtol=1e-3)
        assert isinstance(inv.get_inverse(), MinMaxScale)


class TestChannel:
    """Tests for ChannelExpand / ChannelCollapse."""

    @pytest.mark.parametrize(
        "channel_first, shape, expected",
        [
            (True, (8, 8), (1, 1, 8, 8)),  # 2D gains batch + channel
            (False, (8, 8), (1, 8, 8, 1)),
            (True, (3, 8, 8), (3, 1, 8, 8)),  # 3D gains a channel
            (False, (3, 8, 8), (3, 8, 8, 1)),
        ],
    )
    def test_expand_shapes(self, channel_first, shape, expected):
        """The channel dim is inserted first/last for 2D and 3D inputs."""
        out = ChannelExpand(channel_first).transform(torch.rand(*shape))
        assert out.shape == expected

    @pytest.mark.parametrize("channel_first, dim", [(True, 1), (False, 3)])
    def test_replicate_shape_and_content(self, channel_first, dim):
        """replicate>1 duplicates the channel and every copy is identical."""
        out = ChannelExpand(channel_first, replicate=3).transform(torch.rand(2, 8, 8))
        assert out.shape[dim] == 3
        c0 = out.index_select(dim, torch.tensor([0]))
        for k in (1, 2):
            assert torch.equal(out.index_select(dim, torch.tensor([k])), c0)

    @pytest.mark.parametrize("channel_first", [True, False])
    @pytest.mark.parametrize("replicate", [0, 3])
    def test_expand_revert_round_trip_3d(self, channel_first, replicate):
        """revert() undoes transform() for 3D inputs (with/without replicate)."""
        x = torch.rand(3, 8, 8)
        t = ChannelExpand(channel_first, replicate=replicate)
        assert torch.equal(t.revert(t.transform(x)), x)

    def test_expand_2d_revert_removes_only_channel(self):
        """2D expand adds batch+channel; revert removes only the channel dim."""
        t = ChannelExpand(True)
        assert t.revert(t.transform(torch.rand(8, 8))).shape == (1, 8, 8)

    @pytest.mark.parametrize("channel_first", [True, False])
    def test_collapse_inverts_expand(self, channel_first):
        """ChannelCollapse.transform undoes ChannelExpand.transform."""
        x = torch.rand(3, 8, 8)
        expanded = ChannelExpand(channel_first).transform(x)
        assert torch.equal(ChannelCollapse(channel_first).transform(expanded), x)

    @pytest.mark.parametrize("channel_first", [True, False])
    def test_collapse_round_trip(self, channel_first):
        """ChannelCollapse.revert re-expands what its transform collapsed."""
        expanded = ChannelExpand(channel_first).transform(torch.rand(3, 8, 8))
        c = ChannelCollapse(channel_first)
        assert torch.equal(c.revert(c.transform(expanded)), expanded)

    def test_standalone_forward(self):
        """Works as a plain v2 transform via __call__."""
        assert ChannelExpand(True)(torch.rand(3, 8, 8)).shape == (3, 1, 8, 8)


class TestSequentialTransform:
    """Tests for SequentialTransform composition."""

    def test_log_normalize_round_trip(self):
        """Log + Affine composes the old log-normalization and inverts cleanly."""
        x = torch.rand(8, 8) * 100 + 1.0
        seq = SequentialTransform(
            LogTransform(), Affine(a=torch.log(x.max()), b=torch.log(x.min()))
        )
        params = seq.make_params([x])
        out = seq.transform(x, params)
        assert torch.allclose(seq.revert(out), x, atol=1e-3, rtol=1e-3)

    def test_get_inverse_round_trip(self):
        """get_inverse yields the reversed chain of child inverses."""
        x = torch.rand(8, 8) * 100 + 1.0
        seq = SequentialTransform(HEPTransform(4, x.min(), x.max()), Affine(2.0, 0.5))
        inv = seq.get_inverse()
        fwd = seq.transform(x, seq.make_params([x]))
        back = inv.transform(fwd, inv.make_params([fwd]))
        assert torch.allclose(back, x, atol=1e-3, rtol=1e-3)

    def test_accepts_list(self):
        """A single list of transforms is accepted like varargs."""
        seq = SequentialTransform([LogTransform(), Affine(2.0, 0.0)])
        assert len(seq.transforms) == 2

    def test_shared_params_across_leaves(self):
        """One make_params draw drives the same crop+flip on every leaf."""
        seq = SequentialTransform(RandomCropND((4, 4, 4)), RandomFlipND(3))
        x = torch.arange(6 * 6 * 6).float().reshape(6, 6, 6)
        params = seq.make_params([x])
        a = seq.transform(x, params)
        b = seq.transform(x.clone(), params)
        assert a.shape == (4, 4, 4)
        assert torch.equal(a, b)

    def test_later_child_sees_cropped_shape(self):
        """A crop before a flip: params thread through the intermediate shape."""
        seq = SequentialTransform(RandomCropND((4, 4, 4)), RandomFlipND(3, p=1.0))
        x = torch.rand(8, 8, 8)
        out = seq.transform(x, seq.make_params([x]))
        assert out.shape == (4, 4, 4)

    def test_works_in_sequence_collate(self):
        """SequentialTransform plugs into SequenceCollate over paired fields."""
        from chuchichaestli.data.collate import SequenceCollate

        seq = SequentialTransform(RandomCropND((4,)), Affine(2.0, 0.0))
        samples = [
            {"xfrac": torch.arange(8).float(), "ionrates": torch.arange(8).float()}
            for _ in range(2)
        ]
        out = SequenceCollate(transform=seq)(samples)
        assert out["xfrac"].shape == (2, 4)
        assert torch.equal(out["xfrac"], out["ionrates"])


class TestZScale:
    """Tests for ZScaleInterval and the ZScale transform."""

    def test_clean_ramp_recovers_full_range(self):
        """A clean ramp has no rejection -> limits collapse to (min, max)."""
        z1, z2 = ZScaleInterval().get_limits(torch.linspace(0, 99, 100))
        assert torch.isclose(z1, torch.tensor(0.0))
        assert torch.isclose(z2, torch.tensor(99.0))

    def test_rejects_bright_outliers(self):
        """Saturated pixels are rejected, so z2 sits well below the max."""
        torch.manual_seed(0)
        bg = torch.randn(4096) * 5 + 100
        bg[:20] = 5000.0
        z1, z2 = ZScaleInterval().get_limits(bg)
        assert z2 < 1000.0  # the 5000-count star was rejected
        assert z1 < z2

    def test_too_few_pixels_returns_min_max(self):
        """Below min_npixels the fit is skipped and (min, max) is returned."""
        x = torch.tensor([3.0, 1.0, 2.0])
        z1, z2 = ZScaleInterval(min_npixels=5).get_limits(x)
        assert (z1, z2) == (x.min(), x.max())

    def test_transform_output_range_and_shape(self):
        """ZScale clips+rescales into feature_range and preserves shape."""
        img = torch.randn(3, 32, 32) * 10 + 50
        t = ZScale(feature_range=(0.0, 1.0))
        out = t.transform(img, t.make_params([img]))
        assert out.shape == (3, 32, 32)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_not_invertible(self):
        """ZScale exposes no revert/get_inverse (lossy + data-dependent)."""
        t = ZScale()
        assert not hasattr(t, "revert")
        assert not hasattr(t, "get_inverse")

    def test_shared_limits_in_sequence_collate(self):
        """One set of (z1, z2) is applied to every paired field of the batch."""
        from chuchichaestli.data.collate import SequenceCollate

        torch.manual_seed(0)
        a = torch.randn(64) * 10 + 50
        samples = [{"a": a, "b": a.clone()} for _ in range(2)]
        out = SequenceCollate(transform=ZScale())(samples)
        assert torch.equal(out["a"], out["b"])

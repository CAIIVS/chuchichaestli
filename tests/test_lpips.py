# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the LPIPS module."""

import torch
import pytest
from chuchichaestli.metrics.lpips import (
    LPIPSLoss,
    LPIPSVGG16,
    LPIPSAlexNet,
    LPIPSSqueezeNet,
    LPIPSResNet18,
    LPIPSConvNeXt,
    LPIPSViT,
    LPIPSSwinV2,
    LPIPSEmbedding,
    LPIPSNonEmbedding,
)


@pytest.fixture(scope="module")
def sample_input():
    """Image batch samples."""
    return torch.rand((2, 3, 64, 64))


@pytest.mark.parametrize(
    "model_cls",
    [
        LPIPSVGG16,
        LPIPSAlexNet,
        LPIPSSqueezeNet,
        LPIPSResNet18,
        LPIPSConvNeXt,
        LPIPSViT,
        LPIPSSwinV2,
    ],
)
def test_feature_extractor_shapes(model_cls, sample_input):
    """Test `FeatureExtractor` initializations and forward methods."""
    model = model_cls()
    model.eval()
    with torch.no_grad():
        features = model(sample_input)
    assert isinstance(features, list)
    assert all(isinstance(f, torch.Tensor) for f in features)
    assert all(f.ndim == 4 for f in features)


@pytest.mark.parametrize(
    "model_cls",
    [
        LPIPSVGG16,
        LPIPSAlexNet,
        LPIPSSqueezeNet,
        LPIPSResNet18,
        LPIPSConvNeXt,
        LPIPSViT,
        LPIPSSwinV2,
    ],
)
def test_feature_extractor_non_default_transform(model_cls, sample_input):
    """Test `FeatureExtractor` case: use_default_transforms=False."""
    model = model_cls(use_default_transforms=False)
    model.eval()
    with torch.no_grad():
        features = model(sample_input)
    assert isinstance(features, list)
    assert all(isinstance(f, torch.Tensor) for f in features)
    assert all(f.ndim == 4 for f in features)


def test_feature_extractor_weights_as_string(sample_input):
    """Test `FeatureExtractor` case: weights='IMAGENET1K_V1'."""
    model = LPIPSVGG16(weights="IMAGENET1K_V1")
    model.eval()
    with torch.no_grad():
        features = model(sample_input)
    assert isinstance(features, list)
    assert all(isinstance(f, torch.Tensor) for f in features)
    assert all(f.ndim == 4 for f in features)


def test_feature_extractor_weights_as_invalid_string():
    """Test `FeatureExtractor` case: invalid weights leads to ValueError."""
    with pytest.raises(KeyError):
        LPIPSVGG16(weights="invalid_weights")


def test_feature_extractor_2d_tensor():
    """Test `FeatureExtractor` case: use_default_transforms=False."""
    sample_2d_input = torch.rand((64, 64))
    model = LPIPSVGG16()
    model.eval()
    with torch.no_grad():
        features = model(sample_2d_input)
    assert isinstance(features, list)
    assert all(isinstance(f, torch.Tensor) for f in features)
    assert all(f.ndim == 4 for f in features)


def test_embedding_block_io(sample_input):
    """Test embedding initialization and forward method."""
    model = LPIPSVGG16()
    features = model(sample_input)
    emb = LPIPSEmbedding(model.feature_channels)
    out = emb(features)
    assert len(out) == len(features)
    assert all(f.shape[1] == 1 for f in out)


def test_non_embedding_block_io(sample_input):
    """Test embedding alternative initialization and forward method."""
    model = LPIPSVGG16()
    features = model(sample_input)
    emb = LPIPSNonEmbedding(model.feature_channels)
    out = emb(features)
    assert len(out) == len(features)
    assert all(f.shape[1] == 1 for f in out)


def test_embedding_out_channels_as_tuple(sample_input):
    """Test embedding alternative initialization and forward method."""
    model = LPIPSVGG16()
    features = model(sample_input)
    emb = LPIPSEmbedding(model.feature_channels, [1] * len(model.feature_channels))
    out = emb(features)
    assert len(out) == len(features)
    assert all(f.shape[1] == 1 for f in out)


def test_embedding_clamp_weights(sample_input):
    """Test embedding alternative initialization and forward method."""
    model = LPIPSVGG16()
    emb = LPIPSEmbedding(model.feature_channels, clamp_weights=True)
    for module in emb.modules():
        if isinstance(module, torch.nn.Conv2d):
            assert torch.all(torch.ge(module.weight.data, 0))


@pytest.mark.parametrize("embedding", [True, False, None])
def test_lpips_forward_runs(sample_input, embedding):
    """Test `LPIPSLoss` initialization and forward method."""
    x1, x2 = sample_input[0:1], sample_input[1:2]
    model = LPIPSLoss(model="vgg", embedding=embedding)
    loss = model(x1, x2)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


def test_lpips_scores_output(sample_input):
    """Test `LPIPSLoss.forward` case: `as_scores=True`."""
    x1, x2 = sample_input[0:1], sample_input[1:2]
    fe = LPIPSVGG16()
    model = LPIPSLoss(model=fe, embedding=LPIPSEmbedding(fe.feature_channels))
    scores = model(x1, x2, as_scores=True)
    assert isinstance(scores, list)
    assert all(s.ndim == 0 for s in scores)


@pytest.mark.parametrize("reduction", [torch.mean, torch.sum, None])
def test_lpips_reductions(reduction, sample_input):
    """Test `LPIPSLoss.forward` case: `reduction=True`."""
    x1, x2 = sample_input[0:1], sample_input[1:2]
    model = LPIPSLoss()
    loss = model(x1, x2, reduction=reduction)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_lpips_repr():
    """Test `LPIPSLoss.__repr__`."""
    model = LPIPSLoss("vgg")
    rep = repr(model)
    assert "LPIPSLoss" in rep
    assert "VGG" in rep


def test_lpips_input_shape_mismatch():
    """Test `LPIPSLoss.forward` case: ValueError."""
    model = LPIPSLoss("vgg")
    x1 = torch.rand(1, 3, 64, 64)
    x2 = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError):
        _ = model(x1, x2)


def test_lpips_save_and_load(tmp_path):
    """Test `LPIPSEmbedding.{save, load}` method."""
    model = LPIPSVGG16()
    emb = LPIPSEmbedding(model.feature_channels)
    filepath = tmp_path / "weights.pth"
    emb.save(filepath)
    emb2 = LPIPSEmbedding(model.feature_channels)
    emb2.load(filepath)
    for p1, p2 in zip(emb.parameters(), emb2.parameters()):
        assert torch.equal(p1, p2)


def test_lpips_pretrained_weights(tmp_path):
    """Test `LPIPSEmbedding` case: pretrained_weights."""
    model = LPIPSVGG16()
    emb = LPIPSEmbedding(model.feature_channels, clamp_weights=True)
    filepath = tmp_path / "weights.pth"
    emb.save(filepath)
    model = LPIPSLoss(model, pretrained_weights=filepath, finetune=True)
    for module in model.embedding.modules():
        if isinstance(module, torch.nn.Conv2d):
            assert torch.all(torch.ge(module.weight.data, 0))


def test_non_embedding_save_and_load(tmp_path):
    """Test `LPIPSNonEmbedding.{save, load}` method."""
    model = LPIPSVGG16()
    emb = LPIPSNonEmbedding(model.feature_channels)
    file = tmp_path / "non_weights.pth"
    emb.save(file)
    emb2 = LPIPSNonEmbedding(model.feature_channels)
    emb2.load(file)
    # parameters should match
    for p1, p2 in zip(emb.parameters(), emb2.parameters()):
        assert torch.equal(p1, p2)


def test_fallback_to_default_model():
    """Test LPIPSLoss fallback to default."""
    model = LPIPSLoss("nonexistent_model")
    assert isinstance(model.model, LPIPSVGG16)
    model = LPIPSLoss()
    assert isinstance(model.model, LPIPSVGG16)


@pytest.mark.parametrize(
    "model_name",
    [
        "vgg16",
        "vgg",
        "alexnet",
        "resnet18",
        "resnet",
        "squeezenet",
        "convnext",
        "vit",
        "swinv2",
    ],
)
def test_model_string_init(model_name):
    """Test LPIPSLoss initialization by string."""
    model = LPIPSLoss(model_name)
    assert isinstance(model, LPIPSLoss)
    assert model_name in model.model.__class__.__name__.lower()


def test_clamp_weights_noop():
    """Test `LPIPSEmbedding.clamp_weights`."""
    emb = LPIPSNonEmbedding([32, 64])
    emb.clamp_weights()  # Should not throw


def test_embedding_load_missing_file(tmp_path):
    """Test LPIPSEmbedding.load case: not path.exists()."""
    emb = LPIPSEmbedding([32, 64])
    emb.load(tmp_path / "not_there.pth")  # Should silently skip or not crash


def test_non_embedding_load_missing_file(tmp_path):
    """Test LPIPSEmbedding.load case: not path.exists()."""
    emb = LPIPSNonEmbedding([32, 64])
    emb.load(tmp_path / "not_there.pth")  # Should silently skip or not crash


def test_dispatch_utils():
    """Test `LPIPSLoss._*` singledispatched staticmethods."""
    model = LPIPSLoss("vgg")
    x = torch.rand(1, 3, 32, 32)
    out1 = model._normalize(x)
    out2 = model._normalize([x, x])
    out3 = model._upsample(x, size=(64, 64))
    out4 = model._upsample([x, x], size=(64, 64))
    out5 = model._upsample([x, x], size=[(64, 64), (64, 64)])
    out6 = model._image_average(x)
    out7 = model._image_average([x, x])
    assert isinstance(out1, torch.Tensor)
    assert isinstance(out2, list)
    assert isinstance(out3, torch.Tensor)
    assert isinstance(out4, list)
    assert isinstance(out5, list)
    assert isinstance(out6, torch.Tensor)
    assert isinstance(out7, list)


def test_dispatch_invalid_types():
    """Test `LPIPSLoss._*` singledispatched staticmethods with incompatible types."""
    model = LPIPSLoss("vgg")
    with pytest.raises(NotImplementedError):
        model._normalize(42)
    with pytest.raises(NotImplementedError):
        model._upsample("string", size=(64, 64))
    with pytest.raises(NotImplementedError):
        model._image_average({"not": "tensor"})


def test_lpips_with_3d_input():
    """Test LPIPSLoss.forward with 3D inputs."""
    model = LPIPSVGG16()
    emb = LPIPSEmbedding(model.feature_channels, softplus=True)
    lpips = LPIPSLoss(model, emb)
    x_3d_1 = torch.rand((2, 1, 32, 32, 8))
    x_3d_2 = torch.rand((2, 1, 32, 32, 8))
    loss = lpips(x_3d_1, x_3d_2, reduction=None)
    assert loss.shape[0] == 2 * 8
    loss2 = lpips(x_3d_1, x_3d_2, reduction=None, sample=4)
    assert loss2.shape[0] == 2 * 4

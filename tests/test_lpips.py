import pytest
import torch
from torch.nn import Module
from chuchichaestli.debug.memory_usage import CudaMemoryStatsVisitor, MPSMemoryAllocationVisitor
from chuchichaestli.metrics.lpips import LearnedPerceptualImagePatchSimilarity as LPIPS


@pytest.mark.parametrize(
    "model_name",
    [
        "vgg",
        "alex",
    ],
)
@pytest.mark.skipif(not CudaMemoryStatsVisitor.has_cuda() and not MPSMemoryAllocationVisitor.has_mps(), reason="GPU not available")
def test_2D_same_input(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = LPIPS(net_type=model_name).to(device)
    for _ in range(2):
        img = torch.rand(2, 1, 64, 64).to(device)
        # dataloader transform from metric.model.transform lambda x: model_transform(x)
        metric.update(img, img)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


@pytest.mark.parametrize(
    "model_name",
    [
        "vgg",
        "alex",
    ],
)
@pytest.mark.skipif(not CudaMemoryStatsVisitor.has_cuda() and not MPSMemoryAllocationVisitor.has_mps(), reason="GPU not available")
def test_3D_same_input(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = LPIPS(net_type=model_name).to(device)
    for _ in range(2):
        img = torch.rand(2, 1, 2, 64, 64).to(device)
        # dataloader transform from metric.model.transform lambda x: model_transform(x)
        metric.update(img, img)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


@pytest.mark.parametrize(
    "model_name",
    [
        "vgg",
        "alex",
    ],
)
@pytest.mark.skipif(not CudaMemoryStatsVisitor.has_cuda() and not MPSMemoryAllocationVisitor.has_mps(), reason="GPU not available")
def test_2D_same_input_vs_different(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = LPIPS(net_type=model_name).to(device)
    for _ in range(2):
        img = torch.rand(2, 1, 64, 64).to(device)
        metric.update(img, img)

    val_same = metric.compute()
    metric.reset()

    for _ in range(2):
        img1 = torch.rand(2, 1, 64, 64).to(device)
        img2 = torch.rand(2, 1, 64, 64).to(device)
        metric.update(img1, img2)

    val_diff = metric.compute()

    # assert that val_same is smaller than val_diff
    assert torch.all(val_same < val_diff)


@pytest.mark.parametrize(
    "model_name",
    [
        "vgg",
        "alex",
    ],
)
@pytest.mark.skipif(not CudaMemoryStatsVisitor.has_cuda() and not MPSMemoryAllocationVisitor.has_mps(), reason="GPU not available")
def test_3D_same_input_vs_different(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = LPIPS(net_type=model_name).to(device)
    for _ in range(2):
        img = torch.rand(2, 1, 2, 64, 64).to(device)
        metric.update(img, img)

    val_same = metric.compute()
    metric.reset()

    for _ in range(2):
        img1 = torch.rand(2, 1, 2, 64, 64).to(device)
        img2 = torch.rand(2, 1, 2, 64, 64).to(device)
        metric.update(img1, img2)

    val_diff = metric.compute()

    # assert that val_same is smaller than val_diff
    assert torch.all(val_same < val_diff)

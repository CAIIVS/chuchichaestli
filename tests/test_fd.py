import pytest
import torch
from torch.nn import Module
from chuchichaestli.metrics.fd import FrechetDistance


@pytest.mark.parametrize("model_name, num_features", [
    ("inception", 2048),
    ("swav", 2048),
    ("clip", 1024),
    ("dinov2", 1024),
])
def test_2D_same_input(model_name, num_features):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = FrechetDistance(model_name, device, num_features=num_features).to(device)
    for _ in range(2):
        img = torch.rand(2, 1, 64, 64).to(device)
        # dataloader transform from metric.model.transform lambda x: model_transform(x)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert torch.allclose(metric.real_features_sum, metric.fake_features_sum)
    assert torch.allclose(metric.real_features_cov_sum, metric.fake_features_cov_sum)
    assert torch.allclose(metric.real_features_num_samples, metric.fake_features_num_samples)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)

@pytest.mark.parametrize("model_name, num_features", [
    ("inception", 2048),
    ("swav", 2048),
    ("clip", 1024),
    ("dinov2", 1024),
])
def test_3D_same_input(model_name, num_features):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = FrechetDistance(model_name, device, num_features=num_features).to(device)
    for _ in range(2):
        img = torch.rand(2, 1, 32, 64, 64).to(device)
        # dataloader transform from metric.model.transform lambda x: model_transform(x)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert torch.allclose(metric.real_features_sum, metric.fake_features_sum)
    assert torch.allclose(metric.real_features_cov_sum, metric.fake_features_cov_sum)
    assert torch.allclose(metric.real_features_num_samples, metric.fake_features_num_samples)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


if __name__ == "__main__":
    unittest.main()

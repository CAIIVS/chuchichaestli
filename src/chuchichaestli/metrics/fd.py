import torch
from torchmetrics import Metric
from chuchichaestli.metrics.fd import (
    compute_statistics,
    compute_FD_with_reps,
    compute_efficient_FD_with_reps,
)
import numpy as np
from scipy import linalg
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from representations import get_representation
from .models import load_encoder


class FD(Metric):
    
    def __init__(self, model_name: str, device: str):
        # remember to call super
        super().__init__()
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        self.add_state("real", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fake", default=torch.tensor(0), dist_reduce_fx="sum")

        self.model = load_encoder(model_name, device, ckpt=None, arch=None,
                    clean_resize=False, #Use clean resizing (from pillow)
                    sinception=True if model_name=='sinception' else False,
                    depth=0, # Negative depth for internal layers, positive 1 for after projection head.
                    )
        self.device = device

    def compute_FD_with_stats(self, mu1, mu2, sigma1, sigma2, eps=1e-6):
        """
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fd calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        # Return mean and covariance terms and intermediate steps, as well as FD
        mean_term = diff.dot(diff)
        tr1, tr2 = np.trace(sigma1), np.trace(sigma2)
        cov_term = tr1 + tr2 - 2 * tr_covmean

        return mean_term + cov_term

    def update(self, batch, real:bool) -> None:
        
        features = get_representation(self.model, batch, self.device, normalized=False)
        # extract predicted class index for computing accuracy
        if real: 
            self.real += features.sum(dim=0)

        else:
            self.fake += features.sum(dim=0)

    def compute(self) -> torch.Tensor:
        """Compute necessary statistics from representtions"""
        mu_real = np.mean(self.real, axis=0)
        mu_fake = np.mean(self.fake, axis=0)
        sigma_real = np.cov(self.real, rowvar=False)
        sigma_fake = np.cov(self.fake, rowvar=False)

        mu_real = np.atleast_1d(mu_real)
        mu_fake = np.atleast_1d(mu_fake)
        sigma_real = np.atleast_2d(sigma_real)
        sigma_fake = np.atleast_2d(sigma_fake)

        return self.compute_FD_with_stats(mu1, mu2, sigma1, sigma2, eps=eps)



import unittest
import torch
from chuchichaestli.metrics.fd import FD

class TestFD(unittest.TestCase):

    def setUp(self):
        self.model_name = "inception"  # Example model name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fd_metric = FD(self.model_name, self.device)

    def test_2D_same_tensor(self):
        # Create a random 2D tensor
        tensor1 = torch.rand(4, 128, 128).to(self.device)
        
        # Update the metric with the same tensor twice
        self.fd_metric.update(tensor1, real=True)
        self.fd_metric.update(tensor1, real=False)
        
        # Compute the FD score
        score_same = self.fd_metric.compute()
        
        # Reset the metric
        self.fd_metric.reset()
        
        # Create another random 2D tensor
        tensor2 = torch.rand(4, 128, 128).to(self.device)
        
        # Update the metric with two different tensors
        self.fd_metric.update(tensor1, real=True)
        self.fd_metric.update(tensor2, real=False)

        
        # Compute the FD score
        score_diff = self.fd_metric.compute()
        
        # Check that the score for the same tensor is smaller than for different tensors
        self.assertLess(score_same, score_diff, "FD score for the same tensor should be smaller than for different tensors")

    def test_2D_update_and_compute(self):
        # Create a random 2D tensor
        tensor1 = torch.rand(4, 128, 128).to(self.device)
        
        # Update the metric with the same tensor twice
        self.fd_metric.update(tensor1, real=True)
        self.fd_metric.update(tensor1, real=False)
        
        # Compute the FD score
        score = self.fd_metric.compute()
        
        # Check that the score is a float
        self.assertIsInstance(score, float, "FD score should be a float")
        
        # Check that the score is non-negative
        self.assertGreaterEqual(score, 0, "FD score should be non-negative")

if __name__ == "__main__":
    unittest.main()

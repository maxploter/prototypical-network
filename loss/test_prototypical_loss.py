import unittest
import torch
from .prototypical_loss import PrototypicalLoss


class TestPrototypicalLoss(unittest.TestCase):
    def test_forward(self):
        """Test the forward pass of PrototypicalLoss"""
        print("\n=== Starting test_forward ===")

        # Setup: 5-way 1-shot with 5 query samples per class
        k_shot = 1
        n_way = 5
        n_query = 5
        embedding_dim = 64

        print(f"Configuration: {n_way}-way {k_shot}-shot with {n_query} query samples per class")

        # Create loss function
        loss_fn = PrototypicalLoss(k_shot=k_shot)

        # Create dummy embeddings: (n_way * (k_shot + n_query), embedding_dim)
        # Total samples = 5 * (1 + 5) = 30
        n_samples = n_way * (k_shot + n_query)
        embeddings = torch.randn(n_samples, embedding_dim)

        # Create targets: [0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, ...]
        targets = torch.repeat_interleave(torch.arange(n_way), k_shot + n_query)

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Targets: {targets.tolist()}")

        # Call forward
        print("Calling forward...")
        loss = loss_fn(embeddings, targets)

        # Basic assertions
        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0, "Loss should be positive")

        print(f"✓ Loss value: {loss.item():.4f}")
        print(f"✓ Loss shape: {loss.shape}")
        print("=== Test passed! ===\n")

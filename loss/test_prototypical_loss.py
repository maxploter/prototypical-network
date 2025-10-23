import unittest
import torch
from loss.prototypical_loss import PrototypicalLoss


class TestPrototypicalLoss(unittest.TestCase):
    def test_forward(self):
        k_shot = 1
        n_way = 5
        n_query = 5
        embedding_dim = 64

        loss_fn = PrototypicalLoss(k_shot=k_shot)

        n_samples = n_way * (k_shot + n_query)
        embeddings = torch.randn(n_samples, embedding_dim)

        targets = torch.repeat_interleave(torch.arange(n_way), k_shot + n_query)

        loss, accuracy = loss_fn(embeddings, targets)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0, "Loss should be positive")
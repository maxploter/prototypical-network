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

        # Now returns a dictionary
        loss_dict = loss_fn(embeddings, targets)

        # Check that loss_dict is a dictionary
        self.assertIsInstance(loss_dict, dict)

        # Check that required keys are present
        self.assertIn('loss_proto', loss_dict)
        self.assertIn('accuracy', loss_dict)

        # Check loss_proto
        loss = loss_dict['loss_proto']
        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0, "Loss should be positive")

        # Check accuracy
        accuracy = loss_dict['accuracy']
        self.assertIsNotNone(accuracy)
        self.assertTrue(torch.is_tensor(accuracy))
        self.assertGreaterEqual(accuracy.item(), 0.0, "Accuracy should be >= 0")
        self.assertLessEqual(accuracy.item(), 1.0, "Accuracy should be <= 1")

    def test_weight_dict(self):
        """Test that weight_dict is properly defined"""
        k_shot = 1
        loss_fn = PrototypicalLoss(k_shot=k_shot)

        self.assertTrue(hasattr(loss_fn, 'weight_dict'))
        self.assertIn('loss_proto', loss_fn.weight_dict)
        self.assertEqual(loss_fn.weight_dict['loss_proto'], 1.0)

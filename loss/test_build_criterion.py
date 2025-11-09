import unittest
from argparse import Namespace

from loss import build_criterion
from loss.autoencoder_loss import AutoencoderLoss
from loss.prototypical_loss import PrototypicalLoss


class TestBuildCriterion(unittest.TestCase):

  def test_autoencoder_with_different_datasets(self):
    """Test that build_criterion creates AutoencoderLoss with correct loss type for different datasets"""
    test_cases = [
      # (model, dataset_name, dataset_path, expected_loss_type, description)
      ('autoencoder', 'mnist', 'MNIST', 'mse', 'MNIST should use MSE'),
      ('autoencoder', 'tmnist', 'data/tmnist/grayscale', 'mse', 'Grayscale TMNIST should use MSE'),
      ('autoencoder', 'tmnist', 'data/tmnist/thresholded', 'bce', 'Thresholded TMNIST should use BCE'),
      ('autoencoder', 'tmnist', 'data/tmnist/THRESHOLDED', 'bce', 'Uppercase thresholded should use BCE'),
    ]

    for model, dataset_name, dataset_path, expected_loss_type, description in test_cases:
      with self.subTest(description=description):
        args = Namespace(
          model=model,
          dataset_name=dataset_name,
          dataset_path=dataset_path
        )

        criterion = build_criterion(args)

        # Check that it returns AutoencoderLoss
        self.assertIsInstance(criterion, AutoencoderLoss)

        # Check that it uses the correct loss type
        self.assertEqual(criterion.loss_type, expected_loss_type)

        # Check weight_dict has the correct key
        expected_key = f'loss_{expected_loss_type}'
        self.assertIn(expected_key, criterion.weight_dict)
        self.assertEqual(criterion.weight_dict[expected_key], 1.0)

  def test_prototypical_network(self):
    """Test that build_criterion creates PrototypicalLoss for prototypical network"""
    args = Namespace(
      model='prototypical',
      k_shot=5
    )

    criterion = build_criterion(args)

    # Check that it returns PrototypicalLoss
    self.assertIsInstance(criterion, PrototypicalLoss)

    # Check that k_shot is set correctly
    self.assertEqual(criterion.k_shot, 5)

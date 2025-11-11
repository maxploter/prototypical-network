"""
Unit tests for build_metrics module.
"""
import unittest

import torch

from loss.build_metrics import ROCAUCPreprocessor


class TestROCAUCPreprocessor(unittest.TestCase):
  """Test cases for ROCAUCPreprocessor class."""

  def setUp(self):
    """Set up test fixtures."""
    self.preprocessor = ROCAUCPreprocessor()

  def test_preprocessor_with_float_targets(self):
    """Test preprocessor with float target values (0.0 and 1.0)."""
    # This is the typical case for autoencoder reconstructions
    y_pred = torch.tensor([
      [[0.1, 0.85], [0.3, 0.92]],
      [[0.05, 0.78], [0.88, 0.15]]
    ])  # Float predictions

    y = torch.tensor([
      [[0.0, 1.0], [0.0, 1.0]],
      [[0.0, 1.0], [1.0, 0.0]]
    ])  # Float targets (0.0 and 1.0)

    output = (y_pred, y)
    result = self.preprocessor(output)

    self.assertIsNotNone(result)
    preprocessed_pred, preprocessed_target = result

    # Should convert float targets to int
    self.assertEqual(preprocessed_target.dtype, torch.int32)

    # Verify the converted values are correct
    expected_target = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0], dtype=torch.int32)
    self.assertTrue(torch.equal(preprocessed_target, expected_target))

    # Verify predictions are unchanged and flattened
    expected_pred = torch.tensor([0.1, 0.85, 0.3, 0.92, 0.05, 0.78, 0.88, 0.15])
    self.assertTrue(torch.allclose(preprocessed_pred, expected_pred))

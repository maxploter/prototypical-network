"""
Unit tests for build_metrics module.
"""
import unittest

import torch

from loss.build_metrics import ROCAUCPreprocessor, ChessAccuracyPreprocessor


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
    ])  # Float predictions (logits)

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

    # Verify predictions are sigmoid-transformed and flattened
    expected_pred = torch.sigmoid(torch.tensor([0.1, 0.85, 0.3, 0.92, 0.05, 0.78, 0.88, 0.15]))
    self.assertTrue(torch.allclose(preprocessed_pred, expected_pred))

  # NEW TESTS FOR CHESS PREPROCESSORS

  # TESTS FOR CHESS ACCURACY PREPROCESSOR

  def test_chess_accuracy_basic_preprocessing(self):
    """Test ChessAccuracyPreprocessor with valid multi-class chess data."""
    preprocessor = ChessAccuracyPreprocessor()
    batch_size, num_classes, num_positions = 4, 13, 64

    # Create sample data
    logits = torch.randn(batch_size, num_classes, num_positions)
    targets = torch.randint(0, num_classes, (batch_size, num_positions))

    # Preprocess
    y_pred, y = preprocessor((logits, targets))

    # Check shapes - predictions should be (N, C) for multi-class, targets (N,)
    expected_samples = batch_size * num_positions  # 4 * 64 = 256
    self.assertEqual(y_pred.shape, (expected_samples, num_classes))
    self.assertEqual(y.shape, (expected_samples,))

    # Check data types - predictions are float logits, targets are long class indices
    self.assertEqual(y_pred.dtype, torch.float32)
    self.assertEqual(y.dtype, torch.int64)

  def test_chess_accuracy_argmax_application(self):
    """Test that ChessAccuracyPreprocessor returns logits in correct format for Ignite Accuracy."""
    preprocessor = ChessAccuracyPreprocessor()

    # Create logits where each position has a different max class
    logits = torch.zeros(2, 13, 64)

    # Position 0: class 0 has max logit
    logits[:, 0, 0] = 10.0
    # Position 1: class 1 has max logit
    logits[:, 1, 1] = 10.0
    # Position 2: class 2 has max logit
    logits[:, 2, 2] = 10.0

    targets = torch.zeros(2, 64, dtype=torch.long)

    y_pred, y = preprocessor((logits, targets))

    # Check that predictions are logits (not argmax), shape (N, C)
    self.assertEqual(y_pred.shape, (128, 13))  # 2 batches * 64 positions = 128, 13 classes

    # Verify that the argmax of returned logits gives expected classes
    # This verifies Ignite's Accuracy will compute the right thing
    pred_classes = torch.argmax(y_pred, dim=1)
    self.assertEqual(pred_classes[0].item(), 0)  # First position should have class 0 as max
    self.assertEqual(pred_classes[1].item(), 1)  # Second position should have class 1 as max
    self.assertEqual(pred_classes[2].item(), 2)  # Third position should have class 2 as max

  def test_chess_accuracy_prediction_range(self):
    """Test that ChessAccuracyPreprocessor logits produce valid class predictions in range [0, 12]."""
    preprocessor = ChessAccuracyPreprocessor()

    logits = torch.randn(3, 13, 64)
    targets = torch.randint(0, 13, (3, 64))

    y_pred, y = preprocessor((logits, targets))

    # Predictions are logits (N, C), verify argmax produces valid classes
    pred_classes = torch.argmax(y_pred, dim=1)
    self.assertTrue(torch.all(pred_classes >= 0))
    self.assertTrue(torch.all(pred_classes < 13))

  def test_chess_accuracy_perfect_predictions(self):
    """Test ChessAccuracyPreprocessor with perfect predictions (logits match targets)."""
    preprocessor = ChessAccuracyPreprocessor()
    batch_size = 2

    logits = torch.zeros(batch_size, 13, 64)
    targets = torch.randint(0, 13, (batch_size, 64))

    # Set logits so argmax matches targets
    for b in range(batch_size):
      for pos in range(64):
        target_class = targets[b, pos].item()
        logits[b, target_class, pos] = 10.0

    y_pred, y = preprocessor((logits, targets))

    # Predictions are logits, verify argmax matches targets
    pred_classes = torch.argmax(y_pred, dim=1)
    self.assertTrue(torch.all(pred_classes == y))

  def test_chess_accuracy_output_flattening(self):
    preprocessor = ChessAccuracyPreprocessor()

    logits = torch.randn(2, 13, 64)
    targets = torch.tensor([
      [0, 1, 2, 3] + [0] * 60,
      [4, 5, 6, 7] + [1] * 60
    ])

    y_pred, y = preprocessor((logits, targets))

    # Check shapes: predictions (N, C), targets (N,)
    self.assertEqual(y_pred.shape, (128, 13))  # 2 * 64 = 128 samples, 13 classes
    self.assertEqual(y.shape, (128,))

    # Check specific target values are preserved
    self.assertEqual(y[0].item(), 0)
    self.assertEqual(y[1].item(), 1)
    self.assertEqual(y[2].item(), 2)
    self.assertEqual(y[3].item(), 3)
    self.assertEqual(y[64].item(), 4)  # First position of second batch
    self.assertEqual(y[65].item(), 5)

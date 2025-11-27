"""
Unit tests for build_metrics module.
"""
import unittest

import torch

from loss.build_metrics import ROCAUCPreprocessor, ChessROCAUCPreprocessor, ChessAccuracyPreprocessor


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

  # NEW TESTS FOR CHESS PREPROCESSORS

  def test_chess_roc_auc_basic_preprocessing(self):
    """Test ChessROCAUCPreprocessor with valid multi-class chess data."""
    preprocessor = ChessROCAUCPreprocessor(num_classes=13)
    batch_size, num_classes, num_positions = 4, 13, 64

    # Create sample data: (B, C, D) where B=4, C=13, D=64
    logits = torch.randn(batch_size, num_classes, num_positions)
    targets = torch.randint(0, num_classes, (batch_size, num_positions))

    # Preprocess
    result = preprocessor((logits, targets))
    self.assertIsNotNone(result)

    y_pred, y = result

    # Check shapes
    expected_samples = batch_size * num_positions  # 4 * 64 = 256
    self.assertEqual(y_pred.shape, (expected_samples, num_classes))
    self.assertEqual(y.shape, (expected_samples,))

    # Check that predictions are probabilities (sum to 1)
    prob_sums = y_pred.sum(dim=1)
    torch.testing.assert_close(prob_sums, torch.ones(expected_samples), rtol=1e-5, atol=1e-5)

    # Check that all probabilities are between 0 and 1
    self.assertTrue(torch.all(y_pred >= 0))
    self.assertTrue(torch.all(y_pred <= 1))

    # Check target data type
    self.assertEqual(y.dtype, torch.long)

  def test_chess_roc_auc_softmax_application(self):
    """Test that ChessROCAUCPreprocessor correctly applies softmax to logits."""
    preprocessor = ChessROCAUCPreprocessor(num_classes=13)

    # Create logits with known values
    logits = torch.zeros(2, 13, 64)
    logits[:, 5, :] = 10.0  # Make class 5 have highest logit

    targets = torch.randint(0, 13, (2, 64))

    result = preprocessor((logits, targets))
    y_pred, y = result

    # After softmax, class 5 should have highest probability for all positions
    predicted_classes = torch.argmax(y_pred, dim=1)
    self.assertTrue(torch.all(predicted_classes == 5))

  def test_chess_roc_auc_output_shape_transformation(self):
    """Test that ChessROCAUCPreprocessor correctly reshapes from (B, C, D) to (B*D, C)."""
    preprocessor = ChessROCAUCPreprocessor(num_classes=13)

    logits = torch.randn(3, 13, 64)
    targets = torch.randint(0, 13, (3, 64))

    result = preprocessor((logits, targets))
    y_pred, y = result

    # Should have 3*64 = 192 samples, each with 13 class probabilities
    self.assertEqual(y_pred.shape, (192, 13))
    self.assertEqual(y.shape, (192,))

  def test_chess_roc_auc_single_class_returns_none(self):
    """Test that ChessROCAUCPreprocessor returns None when only one class is present."""
    preprocessor = ChessROCAUCPreprocessor(num_classes=13)

    logits = torch.randn(2, 13, 64)
    # All targets are class 0
    targets = torch.zeros(2, 64, dtype=torch.long)

    result = preprocessor((logits, targets))
    self.assertIsNone(result)

  def test_chess_roc_auc_multiple_classes_returns_valid(self):
    """Test that ChessROCAUCPreprocessor works when multiple classes are present."""
    preprocessor = ChessROCAUCPreprocessor(num_classes=13)

    logits = torch.randn(2, 13, 64)
    targets = torch.zeros(2, 64, dtype=torch.long)
    targets[0, :32] = 1  # Make half of first batch class 1
    targets[0, 32:48] = 5  # Add another class

    result = preprocessor((logits, targets))
    self.assertIsNotNone(result)

    y_pred, y = result
    # Check that we have the classes we set
    unique_classes = torch.unique(y)
    self.assertTrue(0 in unique_classes)
    self.assertTrue(1 in unique_classes)
    self.assertTrue(5 in unique_classes)

  def test_chess_accuracy_basic_preprocessing(self):
    """Test ChessAccuracyPreprocessor with valid multi-class chess data."""
    preprocessor = ChessAccuracyPreprocessor()
    batch_size, num_classes, num_positions = 4, 13, 64

    # Create sample data
    logits = torch.randn(batch_size, num_classes, num_positions)
    targets = torch.randint(0, num_classes, (batch_size, num_positions))

    # Preprocess
    y_pred, y = preprocessor((logits, targets))

    # Check shapes
    expected_samples = batch_size * num_positions  # 4 * 64 = 256
    self.assertEqual(y_pred.shape, (expected_samples,))
    self.assertEqual(y.shape, (expected_samples,))

    # Check data types
    self.assertEqual(y_pred.dtype, torch.int32)
    self.assertEqual(y.dtype, torch.int32)

  def test_chess_accuracy_argmax_application(self):
    """Test that ChessAccuracyPreprocessor correctly applies argmax over class dimension."""
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

    # Check that argmax worked correctly
    self.assertEqual(y_pred[0].item(), 0)  # First position should predict class 0
    self.assertEqual(y_pred[1].item(), 1)  # Second position should predict class 1
    self.assertEqual(y_pred[2].item(), 2)  # Third position should predict class 2

  def test_chess_accuracy_prediction_range(self):
    """Test that ChessAccuracyPreprocessor predicted classes are in valid range [0, 12]."""
    preprocessor = ChessAccuracyPreprocessor()

    logits = torch.randn(3, 13, 64)
    targets = torch.randint(0, 13, (3, 64))

    y_pred, y = preprocessor((logits, targets))

    # All predictions should be between 0 and 12
    self.assertTrue(torch.all(y_pred >= 0))
    self.assertTrue(torch.all(y_pred < 13))

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

    # All predictions should match targets
    self.assertTrue(torch.all(y_pred == y))

  def test_chess_accuracy_output_flattening(self):
    """Test that ChessAccuracyPreprocessor outputs are correctly flattened."""
    preprocessor = ChessAccuracyPreprocessor()

    logits = torch.randn(2, 13, 64)
    targets = torch.tensor([
      [0, 1, 2, 3] + [0] * 60,
      [4, 5, 6, 7] + [1] * 60
    ])

    y_pred, y = preprocessor((logits, targets))

    # Check shape is flattened: 2 * 64 = 128
    self.assertEqual(y.shape, (128,))

    # Check specific target values are preserved
    self.assertEqual(y[0].item(), 0)
    self.assertEqual(y[1].item(), 1)
    self.assertEqual(y[2].item(), 2)
    self.assertEqual(y[3].item(), 3)
    self.assertEqual(y[64].item(), 4)  # First position of second batch
    self.assertEqual(y[65].item(), 5)

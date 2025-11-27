"""
Builder function for creating metrics based on model type.
"""
import torch
import torch.nn.functional as F
from ignite.metrics import ROC_AUC, Accuracy

from utils import is_thresholded_dataset


class ROCAUCPreprocessor:
  """
  Preprocessor for ROC AUC metric on image reconstruction tasks.
  Flattens image tensors for pixel-wise binary classification evaluation.
  Similar to the pattern used in DETR codebase.
  """

  def __call__(self, output):
    """
    Preprocess the output before passing to ROC AUC metric.

    Args:
        output: Tuple of (predictions, targets) - both are image tensors
                predictions are logits (raw unbounded values)

    Returns:
        Preprocessed (predictions, targets) tuple with flattened tensors
        or None if ROC-AUC cannot be computed (only one class present)
    """
    y_pred, y = output

    # Apply sigmoid to convert logits to probabilities
    # ROC_AUC expects probability estimates, not raw logits (per ignite docs)
    y_pred = torch.sigmoid(y_pred)

    # Flatten image tensors for pixel-wise evaluation
    y_pred = y_pred.flatten()
    y = y.flatten()
    y = y.int()

    return (y_pred, y)


class AccuracyPreprocessor:
  """
  Preprocessor for Accuracy metric on image reconstruction tasks.
  Flattens image tensors and applies threshold for pixel-wise binary classification.
  """

  def __init__(self, threshold=0.5):
    """
    Args:
        threshold: Threshold value for converting probabilities to binary predictions (default: 0.5)
    """
    self.threshold = threshold

  def __call__(self, output):
    """
    Preprocess the output before passing to Accuracy metric.

    Args:
        output: Tuple of (predictions, targets) - both are image tensors
                predictions are logits (raw unbounded values)

    Returns:
        Preprocessed (predictions, targets) tuple with flattened tensors
    """
    y_pred, y = output

    # Apply sigmoid to convert logits to probabilities
    y_pred = torch.sigmoid(y_pred)

    # Flatten image tensors for pixel-wise evaluation
    y_pred = y_pred.flatten()
    y = y.flatten()

    # Threshold predictions for binary classification
    y_pred = (y_pred > self.threshold).int()
    y = y.int()

    return (y_pred, y)


class ChessROCAUCPreprocessor:
  """
  Preprocessor for ROC AUC metric on chess dataset with multi-class classification.
  Handles chess board positions with 13 classes (0-12 representing different pieces).
  Converts logits to probabilities and reshapes for position-wise evaluation.
  """

  def __init__(self, num_classes=13):
    """
    Args:
        num_classes: Number of piece classes (default: 13 for chess - empty + 12 piece types)
    """
    self.num_classes = num_classes

  def __call__(self, output):
    """
    Preprocess the output before passing to ROC AUC metric.

    Args:
        output: Tuple of (predictions, targets)
                predictions are logits of shape (B, C, D) where:
                  B = batch size
                  C = 13 classes (piece types)
                  D = 64 positions (chess board squares)
                targets are class indices of shape (B, D) with values 0-12

    Returns:
        Preprocessed (predictions, targets) tuple:
          - predictions: (B*D, C) probability matrix
          - targets: (B*D,) class indices
        or None if ROC-AUC cannot be computed (only one class present)
    """
    y_pred, y = output

    # Apply softmax to convert logits to probabilities for multi-class
    # Shape: (B, C, D) -> (B, C, D)
    y_pred = F.softmax(y_pred, dim=1)

    # Reshape predictions: (B, C, D) -> (B*D, C)
    # This treats each of the 64 board positions as an independent sample
    B, C, D = y_pred.shape
    y_pred = y_pred.permute(0, 2, 1).reshape(-1, C)  # (B*D, C)

    # Flatten targets: (B, D) -> (B*D,)
    if y.dim() == 3 and y.shape[1] == 1:
      y = y.squeeze(1)  # Handle (B, 1, D) -> (B, D)
    y = y.flatten().long()  # (B*D,)

    # Check if we have multiple classes present in this batch
    # ROC-AUC requires at least 2 classes
    unique_classes = torch.unique(y)
    if len(unique_classes) < 2:
      return None  # Cannot compute ROC-AUC with only one class

    return (y_pred, y)


class ChessAccuracyPreprocessor:
  """
  Preprocessor for Accuracy metric on chess dataset with multi-class classification.
  Handles chess board positions with 13 classes (0-12 representing different pieces).
  Computes position-wise accuracy by taking argmax over class dimension.
  """

  def __call__(self, output):
    """
    Preprocess the output before passing to Accuracy metric.

    Args:
        output: Tuple of (predictions, targets)
                predictions are logits of shape (B, C, D) where:
                  B = batch size
                  C = 13 classes (piece types)
                  D = 64 positions (chess board squares)
                targets are class indices of shape (B, D) with values 0-12

    Returns:
        Preprocessed (predictions, targets) tuple with flattened tensors:
          - predictions: (B*D,) predicted class indices
          - targets: (B*D,) ground truth class indices
    """
    y_pred, y = output

    # Take argmax over class dimension to get predicted classes
    # Shape: (B, C, D) -> (B, D)
    y_pred = torch.argmax(y_pred, dim=1)

    # Flatten predictions: (B, D) -> (B*D,)
    y_pred = y_pred.flatten().int()

    # Flatten targets: (B, D) -> (B*D,)
    if y.dim() == 3 and y.shape[1] == 1:
      y = y.squeeze(1)  # Handle (B, 1, D) -> (B, D)
    y = y.flatten().int()

    return (y_pred, y)


class MetricWithPreprocessor:
  """
  Wrapper that combines a metric with a preprocessor.
  Follows the DETR pattern for metric evaluation.
  """

  def __init__(self, metric, preprocessor):
    """
    Args:
        metric: The base metric (e.g., ROC_AUC)
        preprocessor: Preprocessor to apply before metric update
    """
    self.metric = metric
    self.preprocessor = preprocessor

  def reset(self):
    """Reset the underlying metric."""
    self.metric.reset()

  def update(self, output):
    """
    Update metric with preprocessed output.

    Args:
        output: Tuple of (predictions, targets)
    """
    # Preprocess the output
    preprocessed_output = self.preprocessor(output)

    # Skip update if preprocessor returns None (e.g., only one class present)
    if preprocessed_output is None:
      return

    # Update the metric
    self.metric.update(preprocessed_output)

  def compute(self):
    return self.metric.compute()


def build_metrics(args):
    """
    Build metrics dictionary based on the model type.

    Args:
        args: Arguments containing model type and device information

    Returns:
        dict: Dictionary of metrics {metric_name: metric_object} or None if no metrics needed
    """
    metrics = None

    # Metrics are only needed for autoencoder training
    if args.model in ['autoencoder']:
      # Check if this is the chess dataset (multi-class classification)
      if args.dataset_name == 'chess':
        # Chess dataset: multi-class classification with 13 classes
        # Create base metrics
        roc_auc_metric = ROC_AUC(device=args.device)
        accuracy_metric = Accuracy(device=args.device)

        # Create chess-specific preprocessors for multi-class classification
        roc_auc_preprocessor = ChessROCAUCPreprocessor(num_classes=13)
        accuracy_preprocessor = ChessAccuracyPreprocessor()

        # Combine metrics with preprocessors
        metrics = {
          'roc_auc': MetricWithPreprocessor(roc_auc_metric, roc_auc_preprocessor),
          'acc': MetricWithPreprocessor(accuracy_metric, accuracy_preprocessor)
        }
        print(f'Using metrics for {args.model} with chess dataset (13 classes): {list(metrics.keys())}')
        print('Position-wise ROC-AUC and Accuracy metrics will assess chess piece classification performance')

      # Check if we're working with thresholded (binary) data
      elif is_thresholded_dataset(args.dataset_name, args.dataset_path):
        # For thresholded/binary datasets, use pixel-wise ROC-AUC and Accuracy
        # Create base metrics
        roc_auc_metric = ROC_AUC(device=args.device)
        accuracy_metric = Accuracy(device=args.device)

        # Create preprocessors that flatten images for pixel-wise evaluation
        roc_auc_preprocessor = ROCAUCPreprocessor()
        accuracy_preprocessor = AccuracyPreprocessor()

        # Combine metrics with preprocessors
        metrics = {
          'roc_auc': MetricWithPreprocessor(roc_auc_metric, roc_auc_preprocessor),
          'acc': MetricWithPreprocessor(accuracy_metric, accuracy_preprocessor)
        }
        print(f'Using metrics for {args.model} with thresholded data: {list(metrics.keys())}')
        print('Pixel-wise ROC-AUC and Accuracy metrics will assess binary reconstruction performance')

    return metrics

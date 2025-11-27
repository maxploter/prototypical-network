"""
Builder function for creating metrics based on model type.
"""
import torch
from ignite.metrics import ROC_AUC, Accuracy, Precision, Recall, Fbeta, ConfusionMatrix

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


class ChessAccuracyPreprocessor:
  """
  Preprocessor for Accuracy metric on chess dataset with multi-class classification.
  Handles chess board positions with 13 classes (0-12 representing different pieces).
  Reshapes logits and targets for position-wise evaluation.
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
        Preprocessed (predictions, targets) tuple:
          - predictions: (B*D, C) logits for Ignite's multi-class accuracy
          - targets: (B*D,) ground truth class indices
    """
    y_pred, y = output

    # Reshape predictions: (B, C, D) -> (B*D, C)
    # This treats each of the 64 board positions as an independent sample
    B, C, D = y_pred.shape
    y_pred = y_pred.permute(0, 2, 1).reshape(-1, C)  # (B*D, C)

    # Flatten targets: (B, D) -> (B*D,)
    if y.dim() == 3 and y.shape[1] == 1:
      y = y.squeeze(1)  # Handle (B, 1, D) -> (B, D)
    y = y.flatten().long()  # (B*D,) - use .long() for class indices


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
        # Standard metrics for multi-class classification:
        # - Accuracy: Overall correctness
        # - F1-Score: Harmonic mean of precision and recall (macro-averaged)
        # - Confusion Matrix: Per-class performance analysis

        # All Ignite metrics expect:
        # - predictions: (N, C) tensor with raw logits or probabilities
        # - targets: (N,) tensor with class indices
        # Set is_multilabel=False for multi-class classification

        accuracy_metric = Accuracy(is_multilabel=False, device=args.device)

        # For Fbeta to work, Precision and Recall must have average=False
        # Fbeta then does the averaging itself with average=True
        precision_metric = Precision(average=False, is_multilabel=False, device=args.device)
        recall_metric = Recall(average=False, is_multilabel=False, device=args.device)

        # F1-Score (Fbeta with beta=1) with macro averaging
        # This will compute F1 per class and then average them
        f1_metric = Fbeta(beta=1.0, average=True, precision=precision_metric, recall=recall_metric, device=args.device)

        # Confusion Matrix: 13x13 matrix showing predicted vs actual classes
        # Useful for understanding which pieces are confused with each other
        confusion_matrix_metric = ConfusionMatrix(num_classes=13, device=args.device)

        # Create chess-specific preprocessor for multi-class classification
        accuracy_preprocessor = ChessAccuracyPreprocessor()

        # Combine metrics with preprocessor
        metrics = {
          'acc': MetricWithPreprocessor(accuracy_metric, accuracy_preprocessor),
          'f1': MetricWithPreprocessor(f1_metric, accuracy_preprocessor),
          'confusion_matrix': MetricWithPreprocessor(confusion_matrix_metric, accuracy_preprocessor)
        }
        print(f'Using metrics for {args.model} with chess dataset (13 classes): {list(metrics.keys())}')
        print(
          'Position-wise Accuracy, F1-Score (macro-averaged), and Confusion Matrix will assess chess piece classification performance')

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

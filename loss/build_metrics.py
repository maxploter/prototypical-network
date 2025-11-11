"""
Builder function for creating metrics based on model type.
"""
import torch
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
      # Check if we're working with thresholded (binary) data
      is_thresholded = is_thresholded_dataset(args.dataset_name, args.dataset_path)

      if is_thresholded:
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

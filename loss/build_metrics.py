"""
Builder function for creating metrics based on model type.
"""
from ignite.metrics import ROC_AUC

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

    Returns:
        Preprocessed (predictions, targets) tuple with flattened tensors
    """
    y_pred, y = output

    # Flatten image tensors for pixel-wise evaluation
    y_pred = y_pred.flatten()
    y = y.flatten()

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
    # Update the metric
    self.metric.update(preprocessed_output)

  def compute(self):
    """Compute the final metric value."""
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
        # For thresholded/binary datasets, use pixel-wise ROC-AUC
        # Create base metric
        base_metric = ROC_AUC(device=args.device)

        # Create preprocessor that flattens images for pixel-wise evaluation
        preprocessor = ROCAUCPreprocessor()

        # Combine metric with preprocessor
        metrics = {
          'roc_auc': MetricWithPreprocessor(base_metric, preprocessor)
        }
        print(f'Using metrics for {args.model} with thresholded data: {list(metrics.keys())}')
        print('Pixel-wise ROC-AUC metric will assess binary reconstruction performance')

    return metrics

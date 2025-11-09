"""
Builder function for creating metrics based on model type.
"""
from loss.metrics import ROC_AUC

from utils import is_thresholded_dataset


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
        # For thresholded/binary datasets, use ROC-AUC for classification performance
        metrics = {
          'roc_auc': ROC_AUC(device=args.device)
        }
        print(f'Using metrics for {args.model} with thresholded data: {list(metrics.keys())}')
        print('ROC-AUC metric will assess binary classification performance')

    return metrics

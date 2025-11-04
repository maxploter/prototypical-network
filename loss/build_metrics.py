"""
Builder function for creating metrics based on model type.
"""
from ignite.metrics import PSNR


def build_metrics(args):
    """
    Build metrics dictionary based on the model type.

    Args:
        args: Arguments containing model type and device information

    Returns:
        dict: Dictionary of metrics {metric_name: metric_object} or None if no metrics needed
    """
    metrics = None

    # PSNR is only needed for autoencoder training
    if args.model in ['autoencoder']:
        metrics = {
            'psnr': PSNR(data_range=1.0, device=args.device)
        }
        print(f'Using metrics for {args.model}: {list(metrics.keys())}')

    return metrics

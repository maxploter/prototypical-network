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
        # For normalized data with mean=0.1307, std=0.3081:
        # min ≈ (0 - 0.1307) / 0.3081 ≈ -0.424
        # max ≈ (1 - 0.1307) / 0.3081 ≈ 2.821
        # data_range = max - min ≈ 3.245
        data_range = (1.0 - 0.1307) / 0.3081 - (0.0 - 0.1307) / 0.3081

        metrics = {
            'psnr': PSNR(data_range=data_range, device=args.device)
        }
        print(f'Using metrics for {args.model}: {list(metrics.keys())}')
        print(f'PSNR data_range set to {data_range:.4f} for normalized data')

    return metrics

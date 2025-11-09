from loss.autoencoder_loss import AutoencoderLoss
from loss.prototypical_loss import PrototypicalLoss
from utils import is_thresholded_dataset


def build_criterion(args):
  """Build loss criterion with metrics support"""
  if args.model == 'autoencoder':
    # Determine loss type based on dataset
    if args.dataset_name == 'tmnist':
      # Check if dataset is thresholded (binary) or original (grayscale)
      loss_type = 'bce' if is_thresholded_dataset(args.dataset_name, args.dataset_path) else 'mse'
    else:
      # Default to MSE for other datasets (like MNIST)
      loss_type = 'mse'

    return AutoencoderLoss(loss_type=loss_type)
  else:
    return PrototypicalLoss(
      k_shot=args.k_shot
    )

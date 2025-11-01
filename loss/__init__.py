from torch import nn
from loss.prototypical_loss import PrototypicalLoss
from loss.autoencoder_loss import AutoencoderLoss


def build_criterion(args):
  """Build loss criterion with metrics support"""
  if args.model == 'autoencoder':
    return AutoencoderLoss()
  else:
    return PrototypicalLoss(
      k_shot=args.k_shot
    )

from torch import nn
from loss.prototypical_loss import PrototypicalLoss


def build_criterion(args):
  """Build prototypical loss criterion"""
  if args.model == 'autoencoder':
    return nn.MSELoss()
  else:
    return PrototypicalLoss(
      k_shot=args.k_shot
    )

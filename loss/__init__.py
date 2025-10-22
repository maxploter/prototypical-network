from loss.prototypical_loss import PrototypicalLoss


def build_criterion(args):
    """Build prototypical loss criterion"""
    criterion = PrototypicalLoss(
        k_shot=args.k_shot
    )
    return criterion


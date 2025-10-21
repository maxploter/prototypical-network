from .prototype_network_loss import PrototypicalLoss


def build_criterion(args):
    """Build prototypical loss criterion"""
    criterion = PrototypicalLoss()
    return criterion


from model.prototypical_network import PrototypicalNetwork


def build_model(args):
    """Build prototypical network model"""
    model = PrototypicalNetwork(embedding_dim=args.embedding_dim)
    model = model.to(args.device)
    return model


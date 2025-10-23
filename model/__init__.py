from model.prototypical_network import PrototypicalCnnNetwork, PrototypicalAutoencoder
from model.autoencoder import Autoencoder
import torch


def build_model(args):
    if args.model == 'prototypical_cnn':
        model = PrototypicalCnnNetwork(embedding_dim=args.embedding_dim)
    elif args.model == 'prototypical_autoencoder':
        autoencoder = Autoencoder(encoding_dim=64)

        # Load pretrained weights if provided
        if args.autoencoder_path:
            print(f"Loading autoencoder weights from {args.autoencoder_path}")
            checkpoint = torch.load(args.autoencoder_path)

            # Handle different checkpoint formats
            if 'model' in checkpoint:
                autoencoder.load_state_dict(checkpoint['model'])
            else:
                autoencoder.load_state_dict(checkpoint)

            print("Autoencoder weights loaded successfully")

            # Freeze the autoencoder
            for param in autoencoder.parameters():
                param.requires_grad = False
            print("Autoencoder frozen (parameters set to requires_grad=False)")

        model = PrototypicalAutoencoder(encoder=autoencoder.encoder)
    elif args.model == 'autoencoder':
        model = Autoencoder(encoding_dim=args.embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model}.")

    model = model.to(args.device)
    return model

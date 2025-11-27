import torch

from model.autoencoder import Autoencoder
from model.prototypical_network import PrototypicalCnnNetwork, PrototypicalAutoencoder
from utils import is_thresholded_dataset


def build_model(args):
  if args.model == 'prototypical_cnn':
    model = PrototypicalCnnNetwork(embedding_dim=args.embedding_dim)
  elif args.model == 'prototypical_autoencoder':
    # Old models were trained with sigmoid activation (return_logits=False)
    autoencoder = Autoencoder(encoding_dim=args.embedding_dim, return_logits=False)

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

    model = PrototypicalAutoencoder(encoder=autoencoder.encoder, encoder_dim=args.embedding_dim)
  elif args.model == 'autoencoder':
    # Determine whether to return logits or sigmoid output
    return_logits = False  # Default for backward compatibility (sigmoid + MSE)
    num_classes = None  # Default for single value per position
    input_dim = 784  # Default for MNIST (28*28)

    if args.dataset_name == 'tmnist':
      # For thresholded datasets, return logits for BCEWithLogitsLoss
      return_logits = is_thresholded_dataset(args.dataset_name, args.dataset_path)
    elif args.dataset_name == 'chess':
      # Chess uses multi-class classification (13 classes: 0-12)
      return_logits = True
      num_classes = 13
      input_dim = 64  # Chess has 64 squares

    model = Autoencoder(encoding_dim=args.embedding_dim, return_logits=return_logits, num_classes=num_classes,
                        input_dim=input_dim)
  else:
    raise ValueError(f"Unknown model type: {args.model}.")

  model = model.to(args.device)
  return model

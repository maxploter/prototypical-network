"""
Utility functions for loading trained models and datasets.
This is especially useful for interactive environments like Jupyter notebooks or Google Colab.
"""

import argparse

import torch


def load_trained_model(checkpoint_path, device=None):
  """
  Load a trained model from a checkpoint file.

  Args:
      checkpoint_path (str): Path to the checkpoint file (e.g., 'checkpoints/autoencoder_best.pth')
      device (str, optional): Device to load model on ('cuda' or 'cpu').
                              If None, will use cuda if available.

  Returns:
      tuple: (model, args_namespace) where:
          - model: The loaded model ready for inference
          - args_namespace: The argparse.Namespace containing the original training arguments

  Example:
      >>> model, args = load_trained_model('checkpoints/autoencoder_best.pth')
      >>> model.eval()
      >>> # Now you can use the model for inference
  """
  # Lazy import to avoid circular dependency
  from model import build_model

  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

  checkpoint = torch.load(checkpoint_path)

  # Extract args from checkpoint
  if 'args' not in checkpoint:
    raise ValueError(f"Checkpoint at {checkpoint_path} does not contain 'args'. "
                     "This might be an old checkpoint format.")

  # Convert args dict back to Namespace
  args_dict = checkpoint['args']
  args = argparse.Namespace(**args_dict)

  # Override device if specified
  args.device = device

  # Build model using the saved arguments
  model = build_model(args)

  # Load model weights
  model.load_state_dict(checkpoint['model'])
  model.eval()  # Set to evaluation mode by default

  print(f"✓ Model loaded from {checkpoint_path}")
  print(f"  - Model type: {args.model}")
  print(f"  - Device: {device}")
  if 'epoch' in checkpoint:
    print(f"  - Trained for {checkpoint['epoch']} epochs")
  if 'val_loss' in checkpoint:
    print(f"  - Best validation loss: {checkpoint['val_loss']:.4f}")

  return model, args


def load_dataset_from_args(args, split='train'):
  """
  Load a dataset using the arguments from a trained model.

  Args:
      args (argparse.Namespace): Arguments namespace (from load_trained_model)
      split (str): Dataset split to load ('train', 'val', or 'test')

  Returns:
      Dataset: The loaded dataset

  Example:
      >>> model, args = load_trained_model('checkpoints/autoencoder_best.pth')
      >>> train_dataset = load_dataset_from_args(args, split='train')
      >>> val_dataset = load_dataset_from_args(args, split='val')
  """
  # Lazy import to avoid circular dependency
  from dataset import build_dataset

  dataset = build_dataset(args, split=split)
  print(f"✓ Loaded {split} dataset: {len(dataset)} samples")
  return dataset


def load_model_and_dataset(checkpoint_path, split='train', device=None):
  """
  Convenience function to load both model and dataset in one call.

  Args:
      checkpoint_path (str): Path to the checkpoint file
      split (str): Dataset split to load ('train', 'val', or 'test')
      device (str, optional): Device to load model on

  Returns:
      tuple: (model, dataset, args) where:
          - model: The loaded model ready for inference
          - dataset: The loaded dataset
          - args: The argparse.Namespace containing the original training arguments

  Example:
      >>> model, dataset, args = load_model_and_dataset('checkpoints/autoencoder_best.pth')
      >>> # Get a batch
      >>> batch = dataset[0]
      >>> # Run inference
      >>> with torch.no_grad():
      ...     output = model(batch.unsqueeze(0).to(args.device))
  """
  model, args = load_trained_model(checkpoint_path, device=device)
  dataset = load_dataset_from_args(args, split=split)

  return model, dataset, args


def create_args_from_checkpoint(checkpoint_path):
  """
  Extract just the arguments from a checkpoint without loading the model.
  Useful when you want to inspect the training configuration.

  Args:
      checkpoint_path (str): Path to the checkpoint file

  Returns:
      argparse.Namespace: The arguments used during training

  Example:
      >>> args = create_args_from_checkpoint('checkpoints/autoencoder_best.pth')
      >>> print(f"Learning rate: {args.lr}")
      >>> print(f"Embedding dim: {args.embedding_dim}")
  """
  checkpoint = torch.load(checkpoint_path)

  if 'args' not in checkpoint:
    raise ValueError(f"Checkpoint at {checkpoint_path} does not contain 'args'.")

  args = argparse.Namespace(**checkpoint['args'])
  return args

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import build_dataset, build_sampler
from engine import train_one_epoch, evaluate
from loss import build_criterion
from loss.build_metrics import build_metrics
from model import build_model
from utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Training Prototype Network")
    parser.add_argument('--model', type=str, default='prototypical', choices=['prototypical_cnn', 'autoencoder', 'prototypical_autoencoder'],
                        help='Type of model to use: prototypical or autoencoder')
    parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'tmnist'],
                        help='Dataset to use: mnist or tmnist')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset CSV file (required for tmnist). Default: ./data/tmnist/tmnist-glyphs-1812-characters/Glyphs_TMNIST_updated.csv')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class')
    parser.add_argument('--q_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for autoencoder training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--autoencoder_path', type=str, default=None, help='Path to pretrained autoencoder weights')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    # Global seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (controls all random operations)')

    # Dataset reduction arguments
    parser.add_argument('--dataset_reduction', type=float, default=1.0,
                        help='Fraction of training dataset to use (0.0-1.0). Default: 1.0 (use full dataset). '
                             'Example: 0.5 uses 50%% of the data, 0.2 uses 20%%')
    parser.add_argument('--dataset_reduction_strategy', type=str, default='percentage',
                        choices=['percentage', 'class_variability', 'stratified'],
                        help='Strategy for dataset reduction: '
                             'percentage (random sampling), '
                             'class_variability (remove classes), '
                             'stratified (maintain class distribution)')

    # Early stopping and scheduler arguments
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of epochs with no improvement after which training will be stopped. Default: None (no early stopping)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum change in validation loss to qualify as an improvement. Default: 0.0')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler type: step, cosine, plateau, or none')
    parser.add_argument('--lr_drop', type=int, default=5,
                        help='Number of epochs after which learning rate drops (for step scheduler)')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Factor by which the learning rate is reduced (for step and plateau schedulers)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate (for cosine scheduler)')
    return parser.parse_args()

def main(args):
  set_seed(args.seed)

  os.makedirs(args.output_dir, exist_ok=True)
  output_dir = Path(args.output_dir)

  train_dataset = build_dataset(args, split='train')
  train_sampler = build_sampler(args, train_dataset)
  train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

  val_dataset = build_dataset(args, split='val')
  val_sampler = build_sampler(args, val_dataset)
  val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)

  model = build_model(args)

  criterion = build_criterion(args)

  train_metrics = build_metrics(args)
  val_metrics = build_metrics(args)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'Number of trainable parameters: {n_parameters:,}')

  # Create learning rate scheduler
  lr_scheduler = None
  if args.lr_scheduler == 'step':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_gamma)
    print(f'Using StepLR scheduler: lr will drop by factor of {args.lr_gamma} every {args.lr_drop} epochs')
  elif args.lr_scheduler == 'cosine':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    print(f'Using CosineAnnealingLR scheduler: lr will decay to {args.lr_min}')
  elif args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma, patience=2, verbose=True)
    print(f'Using ReduceLROnPlateau scheduler: lr will drop by factor of {args.lr_gamma} when validation loss plateaus')
  else:
    print('No learning rate scheduler used')

  # Training loop for multiple epochs
  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs):
    # Train
    train_loss, train_metrics_dict = train_one_epoch(
        model, criterion, train_dataloader, optimizer, args,
        epoch=epoch + 1, metrics=train_metrics
    )

    # Evaluate
    val_loss, val_loss_dict, val_metrics_dict = evaluate(
        model, criterion, val_dataloader, args,
        epoch=epoch + 1, metrics=val_metrics
    )

    # Get validation accuracy if available (for backward compatibility)
    val_accuracy = val_loss_dict.get('accuracy', 0.0)

    # Prepare log stats
    log_stats = {
        'epoch': epoch + 1,
        'n_parameters': n_parameters,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # Add train metrics
    for k, v in train_metrics_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        log_stats[f'train_{k}'] = v

    # Add validation loss components
    for k, v in val_loss_dict.items():
        log_stats[f'val_{k}'] = v

    # Add validation metrics
    for k, v in val_metrics_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        log_stats[f'val_{k}'] = v

    # Add learning rate
    current_lr = optimizer.param_groups[0]['lr']
    log_stats['lr'] = current_lr

    # Write log stats to file
    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

    # Print epoch summary with all metrics
    print(f'\nEpoch {epoch + 1} Summary:')
    print(f'  Train Loss: {train_loss:.4f}')
    print(f'  Val Loss: {val_loss:.4f}')

    # Print all training metrics
    if train_metrics_dict:
        for k, v in train_metrics_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            print(f'  Train {k}: {v:.4f}')

    # Print all validation loss components
    if val_loss_dict:
        for k, v in val_loss_dict.items():
            print(f'  Val {k}: {v:.4f}')

    # Print all validation metrics
    if val_metrics_dict:
        for k, v in val_metrics_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            print(f'  Val {k}: {v:.4f}')

    # Update learning rate scheduler
    if lr_scheduler is not None:
      if args.lr_scheduler == 'plateau':
        lr_scheduler.step(val_loss)
      else:
        lr_scheduler.step()

      # Log current learning rate
      current_lr = optimizer.param_groups[0]['lr']
      print(f'Current learning rate: {current_lr:.6f}')

    # Check if validation loss improved
    improvement = best_val_loss - val_loss

    # Save best model
    if improvement > args.early_stopping_min_delta:
      best_val_loss = val_loss
      patience_counter = 0
      best_model_path = os.path.join(args.output_dir, args.model + '_best.pth')

      def to_python_type(value):
        """Convert numpy/torch scalars to native Python types."""
        if isinstance(value, torch.Tensor):
          return value.item()
        elif hasattr(value, 'item'):  # numpy scalar
          return value.item()
        elif isinstance(value, dict):
          return {k: to_python_type(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
          return type(value)(to_python_type(v) for v in value)
        else:
          return value

      # Prepare checkpoint data
      checkpoint = {
          'epoch': epoch + 1,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
        'val_loss': float(val_loss),
        'val_loss_dict': to_python_type(val_loss_dict),
        'val_metrics': to_python_type(val_metrics_dict),
          'args': vars(args),
      }

      torch.save(checkpoint, best_model_path)

      # Print save message
      print(f'\n✓ Best model saved to {best_model_path}')
    else:
      patience_counter += 1
      if args.early_stopping_patience is not None:
        print(f'\nNo improvement in validation loss for {patience_counter} epoch(s). Patience: {patience_counter}/{args.early_stopping_patience}')

    # Early stopping check
    if args.early_stopping_patience is not None and patience_counter >= args.early_stopping_patience:
      print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
      print(f'Best validation loss: {best_val_loss:.4f}')
      break

  # Save final model
  model_path = os.path.join(args.output_dir, args.model + '.pth')
  torch.save({
      'model': model.state_dict(),
      'args': vars(args),
  }, model_path)
  print(f'Model saved to {model_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
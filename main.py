from torch.utils.data import DataLoader
import torch
import os
import argparse

from engine import train_one_epoch, evaluate
from dataset import build_dataset, build_sampler
from model import build_model
from loss import build_criterion

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
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of epochs with no improvement after which training will be stopped. Default: None (no early stopping)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum change in validation loss to qualify as an improvement. Default: 0.0')
    return parser.parse_args()

def main(args):
  os.makedirs(args.output_dir, exist_ok=True)

  # Build train dataset and dataloader
  train_dataset = build_dataset(args, split='train')
  train_sampler = build_sampler(args, train_dataset)
  train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

  # Build validation dataset and dataloader
  val_dataset = build_dataset(args, split='val')
  val_sampler = build_sampler(args, val_dataset)
  val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)

  model = build_model(args)

  criterion = build_criterion(args)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Training loop for multiple epochs
  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(args.epochs):
    # Train
    train_loss = train_one_epoch(model, criterion, train_dataloader, optimizer, args, epoch=epoch + 1)

    # Evaluate
    val_loss, val_accuracy = evaluate(model, criterion, val_dataloader, args, epoch=epoch + 1)

    # Check if validation loss improved
    improvement = best_val_loss - val_loss

    # Save best model
    if improvement > args.early_stopping_min_delta:
      best_val_loss = val_loss
      patience_counter = 0
      best_model_path = os.path.join(args.output_dir, args.model + '_best.pth')
      torch.save({
          'epoch': epoch + 1,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'val_loss': val_loss,
          'val_accuracy': val_accuracy,
          'args': vars(args),
      }, best_model_path)
      print(f'Best model saved to {best_model_path} (val_loss: {val_loss:.4f})')
    else:
      patience_counter += 1
      if args.early_stopping_patience is not None:
        print(f'No improvement in validation loss for {patience_counter} epoch(s). Patience: {patience_counter}/{args.early_stopping_patience}')

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
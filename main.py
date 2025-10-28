from torch.utils.data import DataLoader
import torch
import os
import argparse

from engine import train_one_epoch
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
    parser.add_argument('--dataset_split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split to use: train, val, or test')
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
    return parser.parse_args()

def main(args):
  os.makedirs(args.output_dir, exist_ok=True)

  dataset = build_dataset(args)
  sampler = build_sampler(args, dataset)
  dataloader = DataLoader(dataset, batch_sampler=sampler)

  model = build_model(args)

  criterion = build_criterion(args)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Training loop for multiple epochs
  for epoch in range(args.epochs):
    train_one_epoch(model, criterion, dataloader, optimizer, args, epoch=epoch + 1)

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
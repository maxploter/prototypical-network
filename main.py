from torch.utils.data import DataLoader
import torch

from engine import train_one_epoch
from dataset import build_dataset, build_sampler
from model import build_model
from loss import build_criterion
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Prototype Network")
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class')
    parser.add_argument('--q_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main(args):
  dataset = build_dataset(args)
  episode_sampler = build_sampler(args, dataset)
  dataloader = DataLoader(dataset, batch_sampler=episode_sampler)

  model = build_model(args)

  criterion = build_criterion(args)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  train_one_epoch(model, criterion, dataloader, optimizer, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
  """
  Dataset for chess board positions.

  Each sample is a chess board position encoded as 64 values:
    0 = empty
    1-6 = white pieces (P,N,B,R,Q,K)
    7-12 = black pieces (p,n,b,r,q,k)

  Args:
      dataset_path: Path to the CSV file containing board positions
      split: One of 'train', 'val', or 'test'
      transform: Optional transform to apply to board positions
  """

  def __init__(self, dataset_path, split, transform=None):
    assert split in ['train', 'val', 'test'], \
      f"Split must be 'train', 'val', or 'test', got {split}"

    self.dataset_path = Path(dataset_path)
    self.split = split
    self.transform = transform

    # Load the indices for this split
    split_indices = self._load_split_indices()
    print(f"Loading {len(split_indices)} positions for {split} split...")

    # Load only the rows we need
    # Add 1 to indices because CSV has header row
    skiprows = lambda x: x != 0 and (x - 1) not in set(split_indices)
    self.df = pd.read_csv(self.dataset_path, skiprows=skiprows)
    self.df = self.df.reset_index(drop=True)

    print(f"Loaded {split} split: {len(self.df)} positions")

  def _load_split_indices(self):
    """Load the row indices for the current split."""
    dataset_dir = self.dataset_path.parent
    split_file = dataset_dir / f"{self.split}_indices.txt"

    if not split_file.exists():
      raise FileNotFoundError(
        f"Split file not found: {split_file}\n"
        f"Please run prepare_dataset.py first to generate split files."
      )

    indices = np.loadtxt(split_file, dtype=int)
    return indices

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    """
    Returns:
        tuple: (board_tensor, board_tensor) where board_tensor is shape (64,)
               Returns same tensor twice for autoencoder training
    """
    # Get only the 64 square values (sq0 to sq63), excluding move_id column
    square_cols = [f'sq{i}' for i in range(64)]
    board_values = self.df.iloc[idx][square_cols].values.astype(np.float32)

    # Convert to tensor
    board_tensor = torch.from_numpy(board_values)

    # Apply transform if provided
    if self.transform is not None:
      board_tensor = self.transform(board_tensor)

    # For autoencoder training, return the same board as input and target
    return board_tensor, board_tensor

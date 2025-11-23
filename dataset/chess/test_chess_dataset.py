import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.chess import ChessDataset


class TestChessDataset(unittest.TestCase):

  def setUp(self):
    """Create a temporary chess dataset for testing."""
    self.temp_dir = tempfile.mkdtemp()
    self.temp_path = Path(self.temp_dir)

    # Create a small test dataset with 100 positions
    np.random.seed(42)
    positions = []
    for _ in range(100):
      # Random board position (values 0-12)
      pos = np.random.randint(0, 13, size=64)
      positions.append(pos)

    # Save to CSV
    df = pd.DataFrame(positions, columns=[f'sq{i}' for i in range(64)])
    self.csv_path = self.temp_path / 'chess_positions.csv'
    df.to_csv(self.csv_path, index=False)

    # Create split indices
    indices = np.arange(100)
    np.random.shuffle(indices)

    train_indices = indices[:64]
    val_indices = indices[64:80]
    test_indices = indices[80:]

    np.savetxt(self.temp_path / 'train_indices.txt', train_indices, fmt='%d')
    np.savetxt(self.temp_path / 'val_indices.txt', val_indices, fmt='%d')
    np.savetxt(self.temp_path / 'test_indices.txt', test_indices, fmt='%d')

  def test_dataset_creation(self):
    """Test that dataset can be created for each split."""
    for split in ['train', 'val', 'test']:
      dataset = ChessDataset(self.csv_path, split=split)
      self.assertIsInstance(dataset, ChessDataset)

  def test_dataset_lengths(self):
    """Test that dataset splits have correct lengths."""
    train_dataset = ChessDataset(self.csv_path, split='train')
    val_dataset = ChessDataset(self.csv_path, split='val')
    test_dataset = ChessDataset(self.csv_path, split='test')

    self.assertEqual(len(train_dataset), 64)
    self.assertEqual(len(val_dataset), 16)
    self.assertEqual(len(test_dataset), 20)

  def test_getitem(self):
    """Test that __getitem__ returns correct format."""
    dataset = ChessDataset(self.csv_path, split='train')
    board, target = dataset[0]

    # Check types
    self.assertIsInstance(board, torch.Tensor)
    self.assertIsInstance(target, torch.Tensor)

    # Check shape
    self.assertEqual(board.shape, (64,))
    self.assertEqual(target.shape, (64,))

    # For autoencoder, input and target should be the same
    self.assertTrue(torch.equal(board, target))

  def test_board_values(self):
    """Test that board values are in valid range."""
    dataset = ChessDataset(self.csv_path, split='train')

    for i in range(len(dataset)):
      board, _ = dataset[i]
      # All values should be between 0 and 12
      self.assertTrue(torch.all(board >= 0))
      self.assertTrue(torch.all(board <= 12))

  def test_invalid_split(self):
    """Test that invalid split raises error."""
    with self.assertRaises(AssertionError):
      ChessDataset(self.csv_path, split='invalid')

  def test_missing_split_file(self):
    """Test that missing split file raises error."""
    # Remove train indices file
    (self.temp_path / 'train_indices.txt').unlink()

    with self.assertRaises(FileNotFoundError):
      ChessDataset(self.csv_path, split='train')

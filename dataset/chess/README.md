# Chess Dataset

This module provides utilities for preparing and loading chess board positions for autoencoder training.

## Overview

The chess dataset consists of board positions extracted from PGN (Portable Game Notation) files. Each position is
encoded as a 64-element vector representing the 8x8 chess board.

### Board Encoding

Each square is encoded as an integer:

- `0` = empty square
- `1-6` = white pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- `7-12` = black pieces (pawn, knight, bishop, rook, queen, king)

Square indexing follows python-chess convention: a1=0, b1=1, ..., h8=63

## Preparation

### Step 1: Download PGN Data

Download chess games in PGN format. Good sources include:

- [Lichess Database](https://database.lichess.org/) - millions of games
- [FICS Games Database](https://www.ficsgames.org/)

Example:

```bash
# Download a month of Lichess games (compressed)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# Decompress (requires zstd)
zstd -d lichess_db_standard_rated_2024-01.pgn.zst
```

### Step 2: Prepare Dataset

Run the preparation script to extract board positions and create train/val/test splits:

```bash
cd data/chess

# Process first 10,000 games
python prepare_dataset.py \
    --pgn_path /path/to/games.pgn \
    --output_dir ./processed \
    --max_games 10000 \
    --seed 42

# Process all games (may take a while)
python prepare_dataset.py \
    --pgn_path /path/to/games.pgn \
    --output_dir ./processed
```

#### Arguments

- `--pgn_path` (required): Path to input PGN file
- `--output_dir`: Directory for output files (default: current directory)
- `--max_games`: Maximum number of games to process (default: all)
- `--train_ratio`: Training set ratio (default: 0.64)
- `--val_ratio`: Validation set ratio (default: 0.16)
- `--test_ratio`: Test set ratio (default: 0.20)
- `--seed`: Random seed for reproducibility (default: 42)

#### Output Files

The script creates:

- `chess_positions.csv`: All board positions (one per row, 64 columns)
- `train_indices.txt`: Row indices for training set
- `val_indices.txt`: Row indices for validation set
- `test_indices.txt`: Row indices for test set

## Usage

### Basic Usage

```python
from dataset.chess import ChessDataset
from torch.utils.data import DataLoader

# Load dataset
train_dataset = ChessDataset(
  dataset_path='data/chess/processed/chess_positions.csv',
  split='train'
)

# Create dataloader
train_loader = DataLoader(
  train_dataset,
  batch_size=32,
  shuffle=True,
  num_workers=4
)

# Iterate
for batch_idx, (boards, targets) in enumerate(train_loader):
  # boards.shape: (batch_size, 64)
  # targets.shape: (batch_size, 64)
  # For autoencoder training, boards and targets are identical
  pass
```

### With AutoencoderDataset Wrapper

If you want to use the existing AutoencoderDataset wrapper:

```python
from dataset.chess import ChessDataset
from dataset.autoencoder_dataset import AutoencoderDataset

# The ChessDataset already returns (board, board) for autoencoder training
# So you can use it directly without the wrapper
train_dataset = ChessDataset(
  dataset_path='data/chess/processed/chess_positions.csv',
  split='train'
)
```

Note: ChessDataset already implements the autoencoder pattern (returning the same board as input and target), so you
don't need the AutoencoderDataset wrapper.

### Example Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.chess import ChessDataset

# Create datasets
train_dataset = ChessDataset('data/chess/processed/chess_positions.csv', 'train')
val_dataset = ChessDataset('data/chess/processed/chess_positions.csv', 'val')

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Your autoencoder model
model = YourAutoencoder(input_dim=64, latent_dim=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
  model.train()
  for boards, targets in train_loader:
    optimizer.zero_grad()
    reconstructed = model(boards)
    loss = criterion(reconstructed, targets)
    loss.backward()
    optimizer.step()
```

## Testing

Run the test suite:

```bash
python -m pytest dataset/chess/test_chess_dataset.py -v
```

Or using unittest:

```bash
python dataset/chess/test_chess_dataset.py
```

## Performance Considerations

### Why Precompute?

1. **Speed**: Parsing PGN files is slow. Precomputing positions allows fast random access during training.
2. **Consistency**: Same train/val/test split across experiments.
3. **Memory**: CSV format is compact and can be loaded incrementally.

### Dataset Size Estimates

- 1 game ≈ 40 positions (average)
- 1 position = 64 bytes (as uint8) ≈ 256 bytes in CSV
- 10,000 games ≈ 400,000 positions ≈ 100 MB CSV
- 100,000 games ≈ 4,000,000 positions ≈ 1 GB CSV

### Recommendations

For autoencoder training:

- Start with 10,000-50,000 games (400K-2M positions)
- Use `max_games` parameter to limit dataset size initially
- Monitor training loss to determine if more data is needed
- High-quality games (rated 2000+) may give better results than random games

## Example: Quick Start

```bash
# 1. Download sample data (1 month of Lichess games)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
zstd -d lichess_db_standard_rated_2024-01.pgn.zst

# 2. Prepare dataset (use first 50,000 games)
cd data/chess
python prepare_dataset.py \
    --pgn_path lichess_db_standard_rated_2024-01.pgn \
    --output_dir ./processed \
    --max_games 50000

# 3. Train autoencoder (from project root)
# Use your existing training script, just load ChessDataset instead
```


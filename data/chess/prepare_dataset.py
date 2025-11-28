import argparse
from pathlib import Path

import chess  # pip install chess
import chess.pgn
import numpy as np
import pandas as pd
from tqdm import tqdm


def board_to_vec(board):
  """
  Encode board as 64 uint8 values:
    0 = empty
    1..6 = white P,N,B,R,Q,K
    7..12 = black p,n,b,r,q,k
  Square indexing follows python-chess: a1=0 ... h8=63.
  """
  v = [0] * 64
  for sq, pc in board.piece_map().items():
    v[sq] = pc.piece_type + (6 if pc.color == chess.BLACK else 0)
  return v


def move_to_index(move):
  """
  Map a move to an integer in [0, 4095] as from*64 + to.
  Promotion piece is intentionally ignored.
  Accepts either chess.Move or UCI string.
  """
  if isinstance(move, str):
    move = chess.Move.from_uci(move)
  return move.from_square * 64 + move.to_square




def process_pgn_to_csv_streaming(pgn_path, output_path, max_games=None, chunk_size=10000):
  """
  Process PGN file and write positions to CSV in a streaming fashion.
  This avoids loading all games into memory at once.

  Args:
      pgn_path: Path to PGN file
      output_path: Path to save CSV file
      max_games: Maximum number of games to process (None for all)
      chunk_size: Number of positions to accumulate before writing to disk

  Returns:
      Tuple of (total_games, total_positions)
  """
  column_names = [f'sq{i}' for i in range(64)] + ['move_id']
  total_positions = 0
  total_games = 0
  buffer = []

  # Write CSV header
  with open(output_path, 'w') as f:
    f.write(','.join(column_names) + '\n')

  print(f"Processing PGN file: {pgn_path}")
  with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
    pbar = tqdm(desc="Processing games", unit="game")

    while True:
      # Read one game at a time
      game = chess.pgn.read_game(f)
      if game is None:
        break

      # Convert game to positions
      b = chess.Board()

      # Add starting position
      row = board_to_vec(b.copy()) + [-1]
      buffer.append(row)
      total_positions += 1

      # Process each move
      for move in game.mainline_moves():
        b.push(move)
        move_id = move_to_index(move)
        row = board_to_vec(b) + [move_id]
        buffer.append(row)
        total_positions += 1

      total_games += 1
      pbar.update(1)

      # Write chunk if buffer is full
      if len(buffer) >= chunk_size:
        df_chunk = pd.DataFrame(buffer, columns=column_names)
        df_chunk.to_csv(output_path, mode='a', header=False, index=False)
        buffer = []

      # Check max_games limit
      if max_games is not None and total_games >= max_games:
        break

    pbar.close()

  # Write remaining buffer
  if buffer:
    df_chunk = pd.DataFrame(buffer, columns=column_names)
    df_chunk.to_csv(output_path, mode='a', header=False, index=False)

  print(f"Processed {total_games} games")
  print(f"Saved {total_positions} positions to {output_path}")
  return total_games, total_positions


def split_dataset(num_positions, output_dir, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
  """
  Create train/val/test split files with row indices.

  Args:
      num_positions: Total number of positions
      output_dir: Directory to save split files
      train_ratio: Ratio for training set
      val_ratio: Ratio for validation set
      test_ratio: Ratio for test set
  """
  assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
    "Ratios must sum to 1.0"

  indices = np.arange(num_positions)
  np.random.shuffle(indices)

  n_train = int(num_positions * train_ratio)
  n_val = int(num_positions * val_ratio)

  train_indices = indices[:n_train]
  val_indices = indices[n_train:n_train + n_val]
  test_indices = indices[n_train + n_val:]

  output_dir = Path(output_dir)

  # Save indices to text files
  np.savetxt(output_dir / 'train_indices.txt', train_indices, fmt='%d')
  np.savetxt(output_dir / 'val_indices.txt', val_indices, fmt='%d')
  np.savetxt(output_dir / 'test_indices.txt', test_indices, fmt='%d')

  print(f"Split created: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


def create_arg_parser():
  """Create and return argument parser for chess dataset preparation."""
  parser = argparse.ArgumentParser(description='Prepare chess dataset from PGN file')
  parser.add_argument('--pgn_path', type=str, required=True,
                      help='Path to PGN file')
  parser.add_argument('--output_dir', type=str, default='.',
                      help='Output directory for dataset files')
  parser.add_argument('--max_games', type=int, default=None,
                      help='Maximum number of games to process')
  parser.add_argument('--train_ratio', type=float, default=0.64,
                      help='Training set ratio')
  parser.add_argument('--val_ratio', type=float, default=0.16,
                      help='Validation set ratio')
  parser.add_argument('--test_ratio', type=float, default=0.20,
                      help='Test set ratio')
  parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
  return parser


def process(pgn_path, output_dir, max_games=None, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, seed=42):
  """
  Process chess games from PGN file and create dataset.

  Args:
      pgn_path: Path to PGN file
      output_dir: Output directory for dataset files
      max_games: Maximum number of games to process (None for all)
      train_ratio: Ratio for training set
      val_ratio: Ratio for validation set
      test_ratio: Ratio for test set
      seed: Random seed for reproducibility

  Returns:
      Dictionary with paths to created files and statistics
  """
  # Set random seed
  np.random.seed(seed)

  # Create output directory
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Process PGN file directly to CSV using streaming approach
  csv_path = output_dir / 'chess_positions.csv'
  num_games, num_positions = process_pgn_to_csv_streaming(pgn_path, csv_path, max_games=max_games)

  # Create train/val/test splits
  print("Creating train/val/test splits...")
  split_dataset(num_positions, output_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio)

  print(f"\nDataset preparation complete!")
  print(f"Dataset saved to: {csv_path}")
  print(f"Split files saved to: {output_dir}")

  return {
    'csv_path': csv_path,
    'train_indices_path': output_dir / 'train_indices.txt',
    'val_indices_path': output_dir / 'val_indices.txt',
    'test_indices_path': output_dir / 'test_indices.txt',
    'num_games': num_games,
    'num_positions': num_positions,
  }


def main():
  parser = create_arg_parser()
  args = parser.parse_args()

  process(
    pgn_path=args.pgn_path,
    output_dir=args.output_dir,
    max_games=args.max_games,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    seed=args.seed
  )


if __name__ == '__main__':
  main()

import argparse
from pathlib import Path

import chess  # pip install chess
import chess.pgn
import numpy as np
import pandas as pd


def extract_moves_from_pgn(path, max_games=None):
  """
  Return list of games, each game is a list of UCI strings.
  Only the mainline is read; comments/variations are ignored.
  """
  games = []
  with open(path, "r", encoding="utf-8", errors="ignore") as f:
    n = 0
    while True:
      game = chess.pgn.read_game(f)
      if game is None:
        break
      uci = [m.uci() for m in game.mainline_moves()]
      games.append(uci)
      n += 1
      if max_games is not None and n >= max_games:
        break
  return games


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


def uci_games_to_arrays(uci_games):
  """
  For each game:
    - positions: list shape (T, 64), board after each ply
    - move_ids:  list shape (T,), values in [0, 4095]
  Returns two parallel lists of arrays: positions_list, move_ids_list.
  """
  positions_list, move_ids_list = [], []
  for game in uci_games:
    b = chess.Board()
    pos_rows, ids = [], []
    pos_rows.append(board_to_vec(b.copy()))
    for u in game:
      mv = chess.Move.from_uci(u)
      b.push(mv)
      pos_rows.append(board_to_vec(b))  # board AFTER this move
      ids.append(move_to_index(mv))
    if pos_rows:
      positions_list.append(pos_rows)
      move_ids_list.append(ids)
    else:
      positions_list.append([0] * 64)
      move_ids_list.append([0])
  return positions_list, move_ids_list


def save_positions_to_csv(positions_list, move_ids_list, output_path):
  """
  Save all board positions and moves to a CSV file.
  Each row is one board position (64 values) plus the move_id.

  Args:
      positions_list: List of games, each game is a list of 64-element board vectors
      move_ids_list: List of games, each game is a list of move indices
      output_path: Path to save the CSV file
  """
  all_positions = []
  all_move_ids = []

  for game_positions, game_moves in zip(positions_list, move_ids_list):
    # First position has no move (it's the starting position)
    # Subsequent positions have corresponding moves
    all_positions.append(game_positions[0])
    all_move_ids.append(-1)  # -1 indicates no move (starting position)

    for pos, move_id in zip(game_positions[1:], game_moves):
      all_positions.append(pos)
      all_move_ids.append(move_id)

  # Create DataFrame with columns sq0, sq1, ..., sq63, move_id
  df = pd.DataFrame(all_positions, columns=[f'sq{i}' for i in range(64)])
  df['move_id'] = all_move_ids
  df.to_csv(output_path, index=False)
  print(f"Saved {len(all_positions)} positions to {output_path}")
  return len(all_positions)


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


def main():
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

  args = parser.parse_args()

  # Set random seed
  np.random.seed(args.seed)

  # Create output directory
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"Extracting moves from {args.pgn_path}...")
  uci_games = extract_moves_from_pgn(args.pgn_path, max_games=args.max_games)
  print(f"Extracted {len(uci_games)} games")

  print("Converting games to board positions...")
  positions_list, move_ids_list = uci_games_to_arrays(uci_games)

  # Save positions to CSV
  csv_path = output_dir / 'chess_positions.csv'
  num_positions = save_positions_to_csv(positions_list, move_ids_list, csv_path)

  # Create train/val/test splits
  print("Creating train/val/test splits...")
  split_dataset(num_positions, output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio)

  print(f"\nDataset preparation complete!")
  print(f"Dataset saved to: {csv_path}")
  print(f"Split files saved to: {output_dir}")


if __name__ == '__main__':
  main()

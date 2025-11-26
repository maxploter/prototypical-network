import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from data.chess.prepare_dataset import (
  extract_moves_from_pgn,
  board_to_vec,
  move_to_index,
  uci_games_to_arrays,
  process,
)


class TestPrepareDataset(unittest.TestCase):
  def setUp(self):
    """Create a simple PGN file for testing."""
    self.temp_dir = tempfile.mkdtemp()
    self.pgn_path = Path(self.temp_dir) / 'test_games.pgn'

    # Create a simple PGN file with 2 short games
    pgn_content = """[Event "Test Game 1"]
[Site "Test"]
[Date "2025.11.26"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Test Game 2"]
[Site "Test"]
[Date "2025.11.26"]
[Round "2"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1
"""
    with open(self.pgn_path, 'w') as f:
      f.write(pgn_content)

  def tearDown(self):
    """Clean up temporary files."""
    import shutil
    shutil.rmtree(self.temp_dir)

  def test_extract_moves_from_pgn(self):
    """Test extracting moves from PGN file."""
    games = extract_moves_from_pgn(self.pgn_path, max_games=2)

    # Verify games were extracted correctly
    self.assertEqual(len(games), 2)
    self.assertEqual(len(games[0]), 6)  # 3 full moves = 6 half-moves
    self.assertEqual(len(games[1]), 6)

    # Verify first moves are correct
    self.assertEqual(games[0][0], 'e2e4')  # 1. e4
    self.assertEqual(games[0][1], 'e7e5')  # 1... e5
    self.assertEqual(games[0][2], 'g1f3')  # 2. Nf3

    # Verify second game first moves
    self.assertEqual(games[1][0], 'd2d4')  # 1. d4
    self.assertEqual(games[1][1], 'd7d5')  # 1... d5

  def test_board_to_vec(self):
    """Test board encoding to vector."""
    # Test starting position
    board = chess.Board()
    vec = board_to_vec(board)

    # Should have 64 elements
    self.assertEqual(len(vec), 64)

    # White pawns (piece_type=1) on rank 2 (squares 8-15)
    for sq in range(8, 16):
      self.assertEqual(vec[sq], 1, f"Square {sq} should have white pawn (1)")

    # Black pawns (piece_type=1+6=7) on rank 7 (squares 48-55)
    for sq in range(48, 56):
      self.assertEqual(vec[sq], 7, f"Square {sq} should have black pawn (7)")

    # White rooks on a1 (0) and h1 (7)
    self.assertEqual(vec[0], 4)  # White rook = piece_type 4
    self.assertEqual(vec[7], 4)  # White rook

    # Black rooks on a8 (56) and h8 (63)
    self.assertEqual(vec[56], 10)  # Black rook = 4 + 6 = 10
    self.assertEqual(vec[63], 10)  # Black rook

    # Empty squares in the middle
    for sq in range(16, 48):
      self.assertEqual(vec[sq], 0, f"Square {sq} should be empty (0)")

  def test_move_to_index(self):
    """Test move encoding to index."""
    # Test e2e4 (e2 is square 12, e4 is square 28)
    move = chess.Move.from_uci('e2e4')
    idx = move_to_index(move)
    expected = 12 * 64 + 28
    self.assertEqual(idx, expected)

    # Test with UCI string directly
    idx_from_str = move_to_index('e2e4')
    self.assertEqual(idx_from_str, expected)

    # Test another move: d2d4 (d2=11, d4=27)
    idx2 = move_to_index('d2d4')
    expected2 = 11 * 64 + 27
    self.assertEqual(idx2, expected2)

    # Test knight move: g1f3 (g1=6, f3=21)
    idx3 = move_to_index('g1f3')
    expected3 = 6 * 64 + 21
    self.assertEqual(idx3, expected3)

  def test_uci_games_to_arrays(self):
    """Test converting UCI games to position and move arrays."""
    games = extract_moves_from_pgn(self.pgn_path, max_games=2)
    positions_list, move_ids_list = uci_games_to_arrays(games)

    # Should have 2 games
    self.assertEqual(len(positions_list), 2)
    self.assertEqual(len(move_ids_list), 2)

    # First game: 6 moves -> 7 positions (initial + 6 after each move)
    self.assertEqual(len(positions_list[0]), 7)
    self.assertEqual(len(move_ids_list[0]), 6)

    # Second game: 6 moves -> 7 positions
    self.assertEqual(len(positions_list[1]), 7)
    self.assertEqual(len(move_ids_list[1]), 6)

    # Each position should have 64 elements
    for pos in positions_list[0]:
      self.assertEqual(len(pos), 64)

    # Move IDs should be in valid range [0, 4095]
    for move_id in move_ids_list[0]:
      self.assertGreaterEqual(move_id, 0)
      self.assertLess(move_id, 4096)

    # First position should be the starting position
    board = chess.Board()
    expected_start = board_to_vec(board)
    self.assertEqual(positions_list[0][0], expected_start)

    # First move should be e2e4 (12*64 + 28 = 796)
    expected_first_move = move_to_index('e2e4')
    self.assertEqual(move_ids_list[0][0], expected_first_move)

  def test_process(self):
    """Test the complete processing pipeline."""
    # Set up output directory
    output_dir = Path(self.temp_dir) / 'output'

    # Run the process function
    result = process(
      pgn_path=str(self.pgn_path),
      output_dir=str(output_dir),
      max_games=2,
      train_ratio=0.6,
      val_ratio=0.2,
      test_ratio=0.2,
      seed=42
    )

    # Verify result dictionary contains expected keys
    self.assertIn('csv_path', result)
    self.assertIn('train_indices_path', result)
    self.assertIn('val_indices_path', result)
    self.assertIn('test_indices_path', result)
    self.assertIn('num_games', result)
    self.assertIn('num_positions', result)

    # Verify statistics
    self.assertEqual(result['num_games'], 2)
    self.assertEqual(result['num_positions'], 14)  # 2 games * 7 positions each

    # Verify CSV file exists and has correct structure
    csv_path = result['csv_path']
    self.assertTrue(csv_path.exists())

    df = pd.read_csv(csv_path)
    self.assertEqual(len(df), 14)
    self.assertEqual(len(df.columns), 65)  # 64 squares + move_id

    # Verify starting positions have move_id = -1
    self.assertEqual(df.iloc[0]['move_id'], -1)
    self.assertEqual(df.iloc[7]['move_id'], -1)

    # Verify split files exist
    self.assertTrue(result['train_indices_path'].exists())
    self.assertTrue(result['val_indices_path'].exists())
    self.assertTrue(result['test_indices_path'].exists())

    # Load and verify splits
    train_indices = np.loadtxt(result['train_indices_path'], dtype=int)
    val_indices = np.loadtxt(result['val_indices_path'], dtype=int)
    test_indices = np.loadtxt(result['test_indices_path'], dtype=int)

    # Verify splits sum to total positions
    total = len(train_indices) + len(val_indices) + len(test_indices)
    self.assertEqual(total, 14)

    # Verify no overlap between splits
    all_indices = set(train_indices) | set(val_indices) | set(test_indices)
    self.assertEqual(len(all_indices), 14)

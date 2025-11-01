import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from data.tmnist.prepare_dataset import process


class TestPrepareDataset(unittest.TestCase):
  """Unit tests for prepare_dataset.py functions."""

  def setUp(self):
    """Set up test fixtures."""
    np.random.seed(42)
    self.temp_dir = tempfile.TemporaryDirectory()
    self.test_dir = Path(self.temp_dir.name)

  def tearDown(self):
    """Clean up test files."""
    self.temp_dir.cleanup()

  def test_process_creates_split_files(self):
    """Test that process() creates train/val/test split files."""
    # Create a test CSV file with labels
    csv_path = self.test_dir / "test_dataset.csv"
    labels = [str(i) for i in range(20)]
    df = pd.DataFrame({
      'labels': labels * 5,  # 100 rows total with 20 unique labels
      'pixel1': range(100),
      'pixel2': range(100, 200)
    })
    df.to_csv(csv_path, index=False)

    # Run process
    process(self.test_dir, 'kaggle/test_dataset')

    # Check that split files were created
    train_file = self.test_dir / "train_labels.txt"
    val_file = self.test_dir / "val_labels.txt"
    test_file = self.test_dir / "test_labels.txt"

    self.assertTrue(train_file.exists(), "train_labels.txt should be created")
    self.assertTrue(val_file.exists(), "val_labels.txt should be created")
    self.assertTrue(test_file.exists(), "test_labels.txt should be created")

    # Check that files contain the expected number of labels
    with open(train_file, 'r') as f:
      train_labels = f.read().strip().split('\n')
    with open(val_file, 'r') as f:
      val_labels = f.read().strip().split('\n')
    with open(test_file, 'r') as f:
      test_labels = f.read().strip().split('\n')

    # With default ratios (0.64, 0.16, 0.20) and 20 labels
    self.assertEqual(len(train_labels), 12, "Train should have 12 labels (64% of 20)")
    self.assertEqual(len(val_labels), 3, "Val should have 3 labels (16% of 20)")
    self.assertEqual(len(test_labels), 5, "Test should have 5 labels (remaining from 20)")

    # Check no overlap between splits
    all_labels = set(train_labels + val_labels + test_labels)
    self.assertEqual(len(all_labels), 20, "Should have all 20 unique labels across all splits")

  def test_process_with_custom_ratios(self):
    """Test process() with custom split ratios."""
    # Create a test CSV file
    csv_path = self.test_dir / "custom_dataset.csv"
    labels = [f"class_{i}" for i in range(30)]
    df = pd.DataFrame({
      'label': labels * 3,  # 90 rows with 30 unique labels
      'feature1': range(90)
    })
    df.to_csv(csv_path, index=False)

    # Run process with custom ratios
    process(self.test_dir, 'kaggle/custom_dataset',
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # Check split sizes
    train_file = self.test_dir / "train_labels.txt"
    val_file = self.test_dir / "val_labels.txt"
    test_file = self.test_dir / "test_labels.txt"

    with open(train_file, 'r') as f:
      train_labels = f.read().strip().split('\n')
    with open(val_file, 'r') as f:
      val_labels = f.read().strip().split('\n')
    with open(test_file, 'r') as f:
      test_labels = f.read().strip().split('\n')

    self.assertEqual(len(train_labels), 18, "Train should have 18 labels (60% of 30)")
    self.assertEqual(len(val_labels), 6, "Val should have 6 labels (20% of 30)")
    self.assertEqual(len(test_labels), 6, "Test should have 6 labels (20% of 30)")

  def test_process_no_csv_file_raises_error(self):
    """Test that process() raises error when no CSV file exists."""
    empty_dir = self.test_dir / "empty"
    empty_dir.mkdir()

    with self.assertRaises(FileNotFoundError) as context:
      process(empty_dir, 'kaggle/nonexistent_dataset')

    self.assertIn("No CSV file found", str(context.exception))

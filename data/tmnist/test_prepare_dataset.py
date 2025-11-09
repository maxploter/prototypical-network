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
    self._create_test_csv('test_dataset.csv', num_labels=20, rows_per_label=5)

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
    train_labels, val_labels, test_labels = self._read_label_splits()

    # With default ratios (0.64, 0.16, 0.20) and 20 labels
    self.assertEqual(len(train_labels), 12, "Train should have 12 labels (64% of 20)")
    self.assertEqual(len(val_labels), 3, "Val should have 3 labels (16% of 20)")
    self.assertEqual(len(test_labels), 5, "Test should have 5 labels (remaining from 20)")

    # Convert to sets for intersection checks
    train_set = set(train_labels)
    val_set = set(val_labels)
    test_set = set(test_labels)

    # Check that label sets do not intersect with each other
    self.assertEqual(len(train_set & val_set), 0, "Train and val sets should not intersect")
    self.assertEqual(len(train_set & test_set), 0, "Train and test sets should not intersect")
    self.assertEqual(len(val_set & test_set), 0, "Val and test sets should not intersect")

    # Check no overlap between splits
    all_labels = set(train_labels + val_labels + test_labels)
    self.assertEqual(len(all_labels), 20, "Should have all 20 unique labels across all splits")

  def test_process_with_custom_ratios(self):
    """Test process() with custom split ratios."""
    # Create a test CSV file
    self._create_test_csv(
      'custom_dataset.csv',
      num_labels=30,
      rows_per_label=3,
      label_column='label',
      num_pre_label_cols=2,
      num_pixel_cols=200
    )

    # Run process with custom ratios
    process(self.test_dir, 'kaggle/custom_dataset',
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # Check split sizes
    train_labels, val_labels, test_labels = self._read_label_splits()

    self.assertEqual(len(train_labels), 18, "Train should have 18 labels (60% of 30)")
    self.assertEqual(len(val_labels), 6, "Val should have 6 labels (20% of 30)")
    self.assertEqual(len(test_labels), 6, "Test should have 6 labels (20% of 30)")

  def test_process_with_otsu_binarization(self):
    """Test that process() with apply_otsu creates thresholded CSV with correct properties."""
    # Create a test CSV file with known pixel patterns
    label_column = 'labels'
    csv_filename = 'test_otsu_dataset.csv'
    self._create_test_csv(
      csv_filename,
      num_labels=10,
      rows_per_label=5,
      label_column=label_column,
      num_pre_label_cols=2,
      num_pixel_cols=784  # Standard 28x28 image
    )

    # Run process with Otsu binarization enabled
    process(self.test_dir, 'kaggle/test_otsu_dataset')

    # Check that both original and thresholded CSV files exist
    original_csv = self.test_dir / csv_filename
    thresholded_csv = self.test_dir / 'test_otsu_dataset_thresholded.csv'

    self.assertTrue(original_csv.exists(), "Original CSV should exist")
    self.assertTrue(thresholded_csv.exists(), "Thresholded CSV should be created")

    # Read both datasets
    df_original = pd.read_csv(original_csv)
    df_thresholded = pd.read_csv(thresholded_csv)

    # Check that both datasets have the same shape
    self.assertEqual(df_original.shape, df_thresholded.shape,
                     "Original and thresholded datasets should have the same shape")

    # Get all columns before and including the label column
    label_idx = df_original.columns.get_loc(label_column)
    metadata_cols = df_original.columns[:label_idx + 1].tolist()

    # Check that metadata columns (including labels) are identical in both datasets
    for col in metadata_cols:
      pd.testing.assert_series_equal(
        df_original[col],
        df_thresholded[col],
        check_names=True,
        obj=f"Metadata column '{col}' should match in both datasets"
      )

    # Get pixel columns (all columns after the label column)
    pixel_cols = [col for col in df_original.columns if col not in metadata_cols]

    # Check that all values in thresholded dataset pixels are either 0 or 1
    thresholded_pixels = df_thresholded[pixel_cols].values
    unique_values = np.unique(thresholded_pixels)
    self.assertTrue(
      np.all(np.isin(unique_values, [0, 1])),
      f"Thresholded dataset should only contain 0 or 1 values, but found: {unique_values}"
    )

    # Check that 0s in the original dataset remain 0s in the thresholded dataset
    # (thresholding should not change 0s to non-zero values)
    original_pixels = df_original[pixel_cols].values
    zero_mask = (original_pixels == 0)

    # All positions that were 0 in original should be 0 in thresholded
    thresholded_zeros_match = thresholded_pixels[zero_mask]
    self.assertTrue(
      np.all(thresholded_zeros_match == 0),
      "All zero pixels in original dataset should remain zero in thresholded dataset"
    )

    # Additional validation: check that at least some pixels were binarized to 1
    # (to ensure the binarization actually happened and didn't just zero everything)
    num_ones = np.sum(thresholded_pixels == 1)
    self.assertGreater(
      num_ones,
      0,
      "Thresholded dataset should contain at least some 1 values (binarization occurred)"
    )

  def test_process_no_csv_file_raises_error(self):
    """Test that process() raises error when no CSV file exists."""
    empty_dir = self.test_dir / "empty"
    empty_dir.mkdir()

    with self.assertRaises(FileNotFoundError) as context:
      process(empty_dir, 'kaggle/nonexistent_dataset')

    self.assertIn("No CSV file found", str(context.exception))

  def _read_label_splits(self):
    """Read train/val/test label splits from files.

    Returns:
      tuple: (train_labels, val_labels, test_labels) as lists of strings
    """
    train_file = self.test_dir / "train_labels.txt"
    val_file = self.test_dir / "val_labels.txt"
    test_file = self.test_dir / "test_labels.txt"

    with open(train_file, 'r') as f:
      train_labels = f.read().strip().split('\n')
    with open(val_file, 'r') as f:
      val_labels = f.read().strip().split('\n')
    with open(test_file, 'r') as f:
      test_labels = f.read().strip().split('\n')

    return train_labels, val_labels, test_labels

  def _create_test_csv(self, filename, num_labels, rows_per_label=5, label_column='labels', num_pre_label_cols=None, num_pixel_cols=None):
    """Helper function to create a test CSV file with labels.

    Args:
      filename: Name of the CSV file to create
      num_labels: Number of unique labels
      rows_per_label: Number of rows per label
      label_column: Name of the label column (default: 'labels')
      num_pre_label_cols: Number of columns before the label column (default: random 0-3)
      num_pixel_cols: Number of pixel columns after the label (default: random 100-784)

    Returns:
      Path to the created CSV file
    """
    csv_path = self.test_dir / filename
    total_rows = num_labels * rows_per_label

    # Generate labels (characters from TMNIST character set)
    # Using simple numeric labels for testing
    labels = [str(i) for i in range(num_labels)] * rows_per_label

    # Randomly determine number of pre-label and pixel columns if not specified
    if num_pre_label_cols is None:
      num_pre_label_cols = np.random.randint(0, 4)  # 0-3 columns before label
    if num_pixel_cols is None:
      num_pixel_cols = np.random.randint(100, 785)  # 100-784 pixel columns

    data = {}

    # Add pre-label columns (e.g., 'names', 'font_family', etc.)
    pre_label_col_names = ['names', 'font_family', 'font_style', 'font_weight']
    for i in range(num_pre_label_cols):
      col_name = pre_label_col_names[i] if i < len(pre_label_col_names) else f'col_{i}'
      # Generate random font-like names
      data[col_name] = [f'Font_{np.random.randint(0, 100)}' for _ in range(total_rows)]

    # Add label column
    data[label_column] = labels

    # Add pixel columns (pixel values from 0-255)
    for i in range(1, num_pixel_cols + 1):
      data[str(i)] = np.random.randint(0, 256, size=total_rows)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

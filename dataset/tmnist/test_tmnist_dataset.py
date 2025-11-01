import unittest
from random import randint

import torch
import csv
from pathlib import Path
from dataset.tmnist.tmnist_dataset import TMNISTDataset
from torchvision import transforms


class TestTMNISTDataset(unittest.TestCase):
  """Unit tests for TMNISTDataset."""

  @classmethod
  def setUpClass(cls):
    """Set up test fixtures that are used across all test methods."""
    # Create test directory in the same location as the test file
    test_dir = Path(__file__).parent
    cls.csv_file = test_dir / "tmnist.csv"

    # Create CSV file with optional columns, label, and 28*28=784 feature columns
    with open(cls.csv_file, 'w', newline='') as f:
      writer = csv.writer(f)

      n_random_headers = randint(1, 5)

      header = [f'optionalCol{i+1}' for i in range(n_random_headers)]
      header.append('labels')
      header.extend([f'pixel{i}' for i in range(28 * 28)])  # 784 pixel columns
      writer.writerow(header)

      # Create 5 rows with classes: 1, d, e, 3, e
      classes = ['1', 'd', 'e', '3', 'e']
      for idx, class_label in enumerate(classes):
        row = [f"opt_{i}" for i in range(n_random_headers)]
        row.append(class_label)
        # Add 784 random pixel values (0-255)
        row.extend([str((idx * 37 + i * 13) % 256) for i in range(28 * 28)])
        writer.writerow(row)

    # Create train_labels.txt with classes e, 3 (one per line)
    train_labels_file = cls.csv_file.parent / "train_labels.txt"
    with open(train_labels_file, 'w') as f:
      f.write('e\n')
      f.write('3\n')

    cls.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])

  @classmethod
  def tearDownClass(cls):
    """Clean up test files after all tests complete."""
    # Remove created test files
    if cls.csv_file.exists():
      cls.csv_file.unlink()

    train_labels_file = cls.csv_file.parent / "train_labels.txt"
    if train_labels_file.exists():
      train_labels_file.unlink()

  def test_train_split_loads(self):
    """Test that train split loads successfully."""
    dataset = TMNISTDataset(
      dataset_path=self.csv_file,
      split='train',
      transform=self.transform
    )
    self.assertEqual(len(dataset), 3, "Train dataset should have 3 samples")
    self.assertTrue(all(label in [0, 1] for image, label in dataset), "Train dataset labels should map to class indices 0 and 1")

    self.assertEqual(len(dataset.targets), 3, "Targets should have 3 elements")
    self.assertTrue(torch.equal(dataset.targets, torch.tensor([0, 1, 0])), "Train dataset targets should map correctly to class indices")

    # Test that image shape matches MNIST format (1, 28, 28)
    image, label = dataset[0]
    self.assertIsInstance(image, torch.Tensor, "Image should be a torch.Tensor")
    self.assertEqual(image.shape, (1, 28, 28), "Image shape should match MNIST format (1, 28, 28)")

  def test_label_column_detection(self):
    """Test that dataset can handle both 'labels' and 'label' column names."""
    # Test with 'label' (singular) column name
    csv_file_singular = Path(__file__).parent / "tmnist_label_singular.csv"

    # Create CSV file with 'label' column instead of 'labels'
    with open(csv_file_singular, 'w', newline='') as f:
      writer = csv.writer(f)

      # Header with 'label' (singular)
      header = ['font_name', 'glyph_name', 'label']
      header.extend([str(i) for i in range(1, 785)])  # 784 pixel columns
      writer.writerow(header)

      # Create 5 rows with classes: 1, d, e, 3, e
      classes = ['1', 'd', 'e', '3', 'e']
      for idx, class_label in enumerate(classes):
        row = ['font1', f'glyph{idx}', class_label]
        # Add 784 random pixel values (0-255)
        row.extend([str((idx * 37 + i * 13) % 256) for i in range(784)])
        writer.writerow(row)

    try:
      # Create train_labels.txt for this test
      train_labels_file = csv_file_singular.parent / "train_labels.txt"
      with open(train_labels_file, 'w') as f:
        f.write('e\n')
        f.write('3\n')

      # Test that dataset loads successfully with 'label' column
      dataset = TMNISTDataset(
        dataset_path=csv_file_singular,
        split='train',
        transform=self.transform
      )

      self.assertEqual(len(dataset), 3, "Dataset should have 3 samples")
      self.assertEqual(dataset.label_col, 'label', "Dataset should detect 'label' column")
      self.assertEqual(len(dataset.targets), 3, "Targets should have 3 elements")

      # Test that we can get items successfully
      image, label = dataset[0]
      self.assertIsInstance(image, torch.Tensor, "Image should be a torch.Tensor")
      self.assertEqual(image.shape, (1, 28, 28), "Image shape should match MNIST format")

    finally:
      # Clean up
      if csv_file_singular.exists():
        csv_file_singular.unlink()

  def test_missing_label_column_raises_error(self):
    """Test that missing both 'labels' and 'label' columns raises appropriate error."""
    csv_file_no_label = Path(__file__).parent / "tmnist_no_label.csv"

    # Create CSV file without 'label' or 'labels' column
    with open(csv_file_no_label, 'w', newline='') as f:
      writer = csv.writer(f)

      # Header without label column
      header = ['font_name', 'glyph_name']
      header.extend([str(i) for i in range(1, 785)])
      writer.writerow(header)

      # Add one row
      row = ['font1', 'glyph1']
      row.extend([str(i % 256) for i in range(784)])
      writer.writerow(row)

    try:
      # This should raise a ValueError
      with self.assertRaises(ValueError) as context:
        dataset = TMNISTDataset(
          dataset_path=csv_file_no_label,
          split='train',
          transform=self.transform
        )

      # Check that error message mentions both column names
      self.assertIn("Neither 'labels' nor 'label'", str(context.exception))

    finally:
      # Clean up
      if csv_file_no_label.exists():
        csv_file_no_label.unlink()

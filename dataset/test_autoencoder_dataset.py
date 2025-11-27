import unittest

import torch
from torch.utils.data import Dataset

from dataset.autoencoder_dataset import AutoencoderDataset


class MockDataset(Dataset):
  """Mock dataset for testing AutoencoderDataset."""

  def __init__(self, num_samples=10, data_shape=(1, 28, 28), dtype=torch.float32):
    """
    Args:
        num_samples: Number of samples in the dataset
        data_shape: Shape of each data sample
        dtype: Data type of the samples
    """
    self.num_samples = num_samples
    self.data_shape = data_shape
    self.dtype = dtype

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    # Create deterministic data based on idx for reproducible testing
    data = torch.ones(self.data_shape, dtype=self.dtype) * (idx + 1)
    label = idx % 10  # Mock label (0-9)
    return data, label


class TestAutoencoderDataset(unittest.TestCase):
  """Unit tests for AutoencoderDataset."""

  def test_initialization(self):
    """Test that AutoencoderDataset initializes correctly."""
    mock_dataset = MockDataset(num_samples=5)
    ae_dataset = AutoencoderDataset(mock_dataset)

    self.assertEqual(len(ae_dataset), 5, "AutoencoderDataset length should match base dataset")
    self.assertIs(ae_dataset.base_dataset, mock_dataset, "Base dataset should be stored correctly")

  def test_flattening_mnist_style(self):
    """Test that MNIST-style data (1, 28, 28) is flattened correctly."""
    mock_dataset = MockDataset(num_samples=3, data_shape=(1, 28, 28))
    ae_dataset = AutoencoderDataset(mock_dataset)

    input_data, target_data = ae_dataset[0]

    # Check shapes
    expected_size = 1 * 28 * 28
    self.assertEqual(input_data.shape, (expected_size,), f"Input should be flattened to ({expected_size},)")
    self.assertEqual(target_data.shape, (expected_size,), f"Target should be flattened to ({expected_size},)")

    # Check that flattening preserves data
    self.assertTrue(torch.all(input_data == 1.0), "First sample should have all values equal to 1.0")

  def test_flattening_chess_style(self):
    """Test that chess-style data (64,) remains correct shape."""
    mock_dataset = MockDataset(num_samples=5, data_shape=(64,))
    ae_dataset = AutoencoderDataset(mock_dataset)

    input_data, target_data = ae_dataset[0]

    # Check shapes
    self.assertEqual(input_data.shape, (64,), "Input should remain (64,)")
    self.assertEqual(target_data.shape, (64,), "Target should remain (64,)")

  def test_input_output_relationship(self):
    """Test that input and target are properly related."""
    mock_dataset = MockDataset(num_samples=5, data_shape=(1, 28, 28))
    ae_dataset = AutoencoderDataset(mock_dataset)

    for idx in range(len(ae_dataset)):
      input_data, target_data = ae_dataset[idx]

      # Input should be float
      self.assertEqual(input_data.dtype, torch.float32, "Input should be float32")

      # Target should be long (int64) for CrossEntropy loss
      self.assertEqual(target_data.dtype, torch.int64, "Target should be int64 (long)")

      # Target should be the long() version of input
      expected_target = input_data.long()
      self.assertTrue(torch.equal(target_data, expected_target),
                      f"Target should be long() version of input for sample {idx}")

  def test_different_data_types(self):
    """Test that AutoencoderDataset handles different input data types."""
    # Test with float32
    mock_dataset_float32 = MockDataset(num_samples=3, data_shape=(10,), dtype=torch.float32)
    ae_dataset_float32 = AutoencoderDataset(mock_dataset_float32)
    input_data, target_data = ae_dataset_float32[0]

    self.assertEqual(input_data.dtype, torch.float32, "Input should preserve float32 dtype")
    self.assertEqual(target_data.dtype, torch.int64, "Target should be int64")

    # Test with float64
    mock_dataset_float64 = MockDataset(num_samples=3, data_shape=(10,), dtype=torch.float64)
    ae_dataset_float64 = AutoencoderDataset(mock_dataset_float64)
    input_data, target_data = ae_dataset_float64[0]

    self.assertEqual(input_data.dtype, torch.float64, "Input should preserve float64 dtype")
    self.assertEqual(target_data.dtype, torch.int64, "Target should be int64")

  def test_indexing(self):
    """Test that indexing works correctly for all samples."""
    num_samples = 10
    mock_dataset = MockDataset(num_samples=num_samples, data_shape=(1, 5, 5))
    ae_dataset = AutoencoderDataset(mock_dataset)

    for idx in range(num_samples):
      input_data, target_data = ae_dataset[idx]

      # Each sample should have values equal to (idx + 1)
      expected_value = idx + 1
      self.assertTrue(torch.all(input_data == expected_value),
                      f"Sample {idx} should have all values equal to {expected_value}")

  def test_label_ignored(self):
    """Test that original labels are ignored (not returned)."""
    mock_dataset = MockDataset(num_samples=5, data_shape=(1, 28, 28))
    ae_dataset = AutoencoderDataset(mock_dataset)

    result = ae_dataset[0]

    # Should return exactly 2 elements: input and target
    self.assertEqual(len(result), 2, "Should return exactly 2 elements (input, target)")
    self.assertIsInstance(result[0], torch.Tensor, "First element should be a tensor")
    self.assertIsInstance(result[1], torch.Tensor, "Second element should be a tensor")

  def test_multidimensional_flattening(self):
    """Test flattening of various multidimensional shapes."""
    test_shapes = [
      (1, 28, 28),  # MNIST
      (3, 32, 32),  # CIFAR-10
      (64,),  # Chess
      (1, 10, 10),  # Small image
    ]

    for shape in test_shapes:
      mock_dataset = MockDataset(num_samples=2, data_shape=shape)
      ae_dataset = AutoencoderDataset(mock_dataset)

      input_data, target_data = ae_dataset[0]

      expected_size = 1
      for dim in shape:
        expected_size *= dim

      self.assertEqual(input_data.shape, (expected_size,),
                       f"Shape {shape} should flatten to ({expected_size},)")
      self.assertEqual(target_data.shape, (expected_size,),
                       f"Shape {shape} target should flatten to ({expected_size},)")

  def test_empty_base_dataset(self):
    """Test behavior with an empty base dataset."""
    mock_dataset = MockDataset(num_samples=0)
    ae_dataset = AutoencoderDataset(mock_dataset)

    self.assertEqual(len(ae_dataset), 0, "Empty base dataset should result in empty AutoencoderDataset")

  def test_value_preservation(self):
    """Test that data values are preserved correctly during flattening."""

    # Create a dataset with specific values
    class SpecificValueDataset(Dataset):
      def __len__(self):
        return 1

      def __getitem__(self, idx):
        # Create a 2x2 tensor with specific values
        data = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32).unsqueeze(0)  # (1, 2, 2)
        return data, 0

    specific_dataset = SpecificValueDataset()
    ae_dataset = AutoencoderDataset(specific_dataset)

    input_data, target_data = ae_dataset[0]

    # Check that values are preserved in the correct order (row-major)
    expected_input = torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float32)
    expected_target = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    self.assertTrue(torch.equal(input_data, expected_input),
                    "Input values should be preserved in row-major order")
    self.assertTrue(torch.equal(target_data, expected_target),
                    "Target values should be long() version of input")

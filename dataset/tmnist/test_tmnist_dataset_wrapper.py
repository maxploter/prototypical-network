import unittest
import torch
import os
import shutil
import tempfile
from torchvision import transforms


class TestTMNISTDatasetWrapper(unittest.TestCase):

  def test_should_keep_classes_dedeicated_per_split(self):
    """Test dataset initialization with a split file."""
    transform = transforms.Compose([
      transforms.ToTensor(),
    ])

  dataset = TMNISTDatasetWrapper(
    mode='train',
    root=self.mnist_root,
    transform=transform,
    download=True,
    split_file=self.train_split
  )

  # Check that only specified classes are loaded
  self.assertEqual(dataset.classes, [0, 1, 2])
  self.assertEqual(len(dataset.class_to_idx), 3)

  # Check class to index mapping
  self.assertIn(0, dataset.class_to_idx)
  self.assertIn(1, dataset.class_to_idx)
  self.assertIn(2, dataset.class_to_idx)

  # Check that targets are remapped to contiguous indices
  unique_targets = torch.unique(dataset.targets)
  self.assertEqual(len(unique_targets), 3)
  self.assertTrue(all(t < 3 for t in unique_targets))

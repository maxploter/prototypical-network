# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np

class TMNISTDatasetWrapper(data.Dataset):
  """
  TMNIST Dataset Wrapper for Prototypical Networks.

  TMNIST contains 1,812 unique character glyphs from various writing systems.
  This wrapper filters the dataset by specified classes to enable
  class-based splitting (e.g., different classes for train/val/test).

  The dataset is organized by 'flavors' - different versions or subsets
  that may come in separate CSV files.
  """

  def __init__(
    self,
    classes,
    split='train',
    root='.',
    flavor='TMNIST_Data',
    transform=None,
    target_transform=None,
    download=False,
  ):
    """
    Initialize TMNIST Dataset Wrapper.

    Args:
      classes: List of class labels to include in this dataset
      split: 'train', 'val', or 'test' (for identification purposes)
      root: Root directory containing the data folder
      flavor: Dataset flavor name (subdirectory name in data/)
      transform: Optional transform to apply to images
      target_transform: Optional transform to apply to targets
      download: Not used (dataset must be pre-downloaded via prepare_dataset.py)
    """
    super(TMNISTDatasetWrapper, self).__init__()

    self.transform = transform
    self.target_transform = target_transform

    # Default transform if none provided
    if self.transform is None:
      self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
      ])

    # Construct path to the flavor directory
    flavor_dir = os.path.join(root, 'data', flavor)

    # Find CSV file in the flavor directory
    csv_file = None
    if os.path.exists(flavor_dir):
      csv_files = [f for f in os.listdir(flavor_dir) if f.endswith('.csv')]
      if csv_files:
        csv_file = os.path.join(flavor_dir, csv_files[0])

    # Fallback: try old structure for backward compatibility
    if csv_file is None or not os.path.exists(csv_file):
      old_path = os.path.join(root, 'TMNIST', 'TMNIST_Data.csv')
      if os.path.exists(old_path):
        csv_file = old_path
        print(f"Warning: Using old dataset structure at {old_path}")
        print(f"Please run dataset/tmnist/data/prepare_dataset.py to reorganize the dataset")

    # Check if file exists
    if csv_file is None or not os.path.exists(csv_file):
      raise FileNotFoundError(
        f"TMNIST data file not found for flavor '{flavor}' in {flavor_dir}.\n"
        f"Please run the preparation script: cd dataset/tmnist/data && python prepare_dataset.py"
      )

    # Load the CSV file
    print(f"Loading TMNIST data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Extract labels and images
    # TMNIST CSV format: first column is label, rest are pixel values
    if 'names' in df.columns:
      label_col = 'names'
    elif 'label' in df.columns:
      label_col = 'label'
    elif 'labels' in df.columns:
      label_col = 'labels'
    else:
      # Assume first column is label
      label_col = df.columns[0]

    all_labels = df[label_col].values
    # All columns except the label column are pixel values
    pixel_cols = [col for col in df.columns if col != label_col]
    all_images = df[pixel_cols].values

    # Store classes and create mapping
    self.classes = sorted(set(classes))  # Ensure sorted for consistency
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    # Filter dataset by specified classes
    mask = np.isin(all_labels, self.classes)
    self.labels = all_labels[mask]
    self.images = all_images[mask]

    # Remap labels to contiguous indices [0, len(classes)-1]
    self._remap_labels()

    print(f"Loaded {len(self.images)} samples from {len(self.classes)} classes")

  def _remap_labels(self):
    """Remap labels to contiguous indices starting from 0."""
    new_labels = np.zeros_like(self.labels)
    for old_label, new_label in self.class_to_idx.items():
      new_labels[self.labels == old_label] = new_label
    self.labels = new_labels

  def __getitem__(self, idx):
    img = self.images[idx]
    target = int(self.labels[idx])

    # Reshape to 28x28 (TMNIST images are 28x28)
    img = img.reshape(28, 28).astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(img, mode='L')

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.images)

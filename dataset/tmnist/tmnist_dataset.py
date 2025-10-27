import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image


class TMNISTDataset(Dataset):
  """
  Dataset wrapper for TMNIST that filters data by split (train/val/test).

  Args:
    dataset_path: Path to the CSV file containing the dataset
    split: One of 'train', 'val', or 'test'
    transform: Optional transform to apply to images
  """

  def __init__(self, dataset_path, split, transform=None):
    assert split in ['train', 'val', 'test'], \
      f"Split must be 'train', 'val', or 'test', got {split}"

    self.dataset_path = Path(dataset_path)
    self.split = split
    self.transform = transform

    # Load the full dataset
    df_full = pd.read_csv(self.dataset_path)

    # Find the 'labels' column and keep only columns from that point onwards
    if 'labels' not in df_full.columns:
      raise ValueError(f"'labels' column not found in dataset. Available columns: {df_full.columns.tolist()}")

    labels_idx = df_full.columns.get_loc('labels')
    # Keep only columns from 'labels' onwards
    self.df = df_full.iloc[:, labels_idx:]

    # Load the labels for this split
    split_labels = self._load_split_labels()

    # Filter the dataset to only include samples from allowed labels
    self.df = self.df[self.df['labels'].isin(split_labels)]
    self.df = self.df.reset_index(drop=True)

    # Create label mapping (original label -> class index)
    self.label_to_idx = {label: idx for idx, label in enumerate(split_labels)}
    self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    # Create targets field (list of class indices for all samples)
    # This matches the MNIST dataset interface
    self.targets = [self.label_to_idx[label] for label in self.df['labels']]

    print(f"Loaded {split} split: {len(self.df)} samples from {len(split_labels)} classes")

  def _load_split_labels(self):
    """Load the labels for the current split from the corresponding text file."""
    # Get the directory containing the CSV file
    dataset_dir = self.dataset_path.parent
    split_file = dataset_dir / f"{self.split}_labels.txt"

    if not split_file.exists():
      raise FileNotFoundError(
        f"Split file not found: {split_file}\n"
        f"Please run prepare_dataset.py first to generate split files."
      )

    labels = []
    with open(split_file, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line:
          # Keep labels as strings to match the CSV format
          # The CSV 'labels' column contains string values
          labels.append(line)

    return labels

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    # Extract label
    original_label = row['labels']
    class_idx = self.label_to_idx[original_label]

    # Extract image data (all columns after 'labels' are pixel data)
    pixel_columns = [col for col in self.df.columns if col != 'labels']
    image_data = row[pixel_columns].values.astype(np.uint8)

    # Extract image data (all columns after label column are pixel data)
    img_size = int(np.sqrt(len(image_data)))
    image = image_data.reshape(img_size, img_size)

    # Convert to PIL Image for transform compatibility
    image = Image.fromarray(image, mode='L')

    # Apply transforms if any
    if self.transform:
      image = self.transform(image)
    else:
      # Default: convert to tensor
      image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0

    return image, class_idx

  def get_labels(self):
    """Return all class indices in the dataset."""
    return [self.label_to_idx[label] for label in self.df['labels']]

  def get_original_labels(self):
    """Return all original labels in the dataset."""
    return self.df['labels'].tolist()

  def num_classes(self):
    """Return the number of classes in this split."""
    return len(self.label_to_idx)

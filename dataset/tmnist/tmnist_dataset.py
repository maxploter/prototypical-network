from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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

    # First, load just the header to detect the label column
    df_header = pd.read_csv(self.dataset_path, nrows=0)

    # Detect the label column name (either 'labels' or 'label')
    if 'labels' in df_header.columns:
      self.label_col = 'labels'
    elif 'label' in df_header.columns:
      self.label_col = 'label'
    else:
      raise ValueError(f"Neither 'labels' nor 'label' column found in dataset. Available columns: {df_header.columns.tolist()}")

    labels_idx = df_header.columns.get_loc(self.label_col)
    print(f"Using label column: '{self.label_col}' at index {labels_idx}")

    # Load only the label column first to do filtering
    print(f"Loading dataset for filtering...")
    df_labels = pd.read_csv(self.dataset_path, usecols=[self.label_col])

    # Load the labels for this split
    split_labels = self._load_split_labels()
    print(f"Loaded {len(split_labels)} labels")

    # Create label mapping (original label -> class index) FIRST
    # This gives us O(1) lookup for filtering
    self.label_to_idx = {label: idx for idx, label in enumerate(split_labels)}
    self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    # Filter to get row indices
    print(f"Filtering dataset...")
    split_labels_set = set(split_labels)
    mask = df_labels[self.label_col].isin(split_labels_set)
    matching_indices = set(mask.index[mask].tolist())
    print(f"Found {len(matching_indices)} matching rows")

    # Create targets directly from filtered labels
    print(f"Creating targets tensor...")
    filtered_labels = df_labels[mask][self.label_col]
    self.targets = torch.tensor([self.label_to_idx[label] for label in filtered_labels], dtype=torch.long)

    # Now load only the matching rows
    print(f"Loading matching rows...")
    df_full = pd.read_csv(self.dataset_path, skiprows=lambda x: x != 0 and x-1 not in matching_indices)

    # Keep only columns from label column onwards
    self.df = df_full.iloc[:, labels_idx:]
    self.df = self.df.reset_index(drop=True)

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
    original_label = row[self.label_col]
    class_idx = self.label_to_idx[original_label]

    # Extract image data (all columns after 'labels' are pixel data)
    pixel_columns = [col for col in self.df.columns if col != self.label_col]
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
      image = torch.from_numpy(np.array(image)).float().unsqueeze(0)
    return image, class_idx

  def get_labels(self):
    """Return all class indices in the dataset."""
    return [self.label_to_idx[label] for label in self.df[self.label_col]]

  def get_original_labels(self):
    """Return all original labels in the dataset."""
    return self.df[self.label_col].tolist()

  def num_classes(self):
    """Return the number of classes in this split."""
    return len(self.label_to_idx)

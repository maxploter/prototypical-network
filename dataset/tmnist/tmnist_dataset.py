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

    # Detect the label column name (either 'labels' or 'label')
    if 'labels' in df_full.columns:
      self.label_col = 'labels'
    elif 'label' in df_full.columns:
      self.label_col = 'label'
    else:
      raise ValueError(f"Neither 'labels' nor 'label' column found in dataset. Available columns: {df_full.columns.tolist()}")

    labels_idx = df_full.columns.get_loc(self.label_col)
    print(f"Using label column: '{self.label_col}' at index {labels_idx}")
    # Keep only columns from label column onwards
    self.df = df_full.iloc[:, labels_idx:]
    # Load the labels for this split
    split_labels = self._load_split_labels()
    print(f"Loaded {len(split_labels)} labels")

    # Create label mapping (original label -> class index) FIRST
    # This gives us O(1) lookup for filtering
    self.label_to_idx = {label: idx for idx, label in enumerate(split_labels)}
    self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    print(f"Filtering dataset (this may take a moment for large datasets)...")
    # Convert split_labels to a set for faster lookup
    split_labels_set = set(split_labels)

    # Use isin() with a set - pandas optimizes this
    self.df = self.df[self.df[self.label_col].isin(split_labels_set)]
    print(f"Filtering complete. Found {len(self.df)} samples. Resetting index...")
    self.df = self.df.reset_index(drop=True)
    print(f"Index reset complete.")

    print(f"Creating targets tensor...")
    # Create targets field (list of class indices for all samples)
    # This matches the MNIST dataset interface
    self.targets = torch.tensor([self.label_to_idx[label] for label in self.df[self.label_col]], dtype=torch.long)

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
      # Default: convert to tensor
      image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0

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

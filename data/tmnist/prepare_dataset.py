import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def download_kaggle_dataset(output_dir, dataset_name):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Extract the actual dataset filename from dataset_name (e.g., "user/dataset" -> "dataset")
  dataset_filename = dataset_name.split('/')[-1]
  csv_file = output_dir / f"{dataset_filename}.csv"

  # Check if dataset already exists
  if csv_file.exists():
    return output_dir

  try:
    import kaggle
    kaggle.api.dataset_download_files(
      dataset_name,
      path=str(output_dir),
      unzip=True
    )
    return output_dir
  except ImportError:
    raise ImportError("Kaggle API not installed. Install with: pip install kaggle")
  except Exception as e:
    raise RuntimeError(f"Failed to download dataset: {e}")


def get_unique_labels(csv_path):
  df = pd.read_csv(csv_path)
  labels = df['label'].unique()
  unique_labels = sorted(list(labels))
  return unique_labels


def split_labels(labels, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
  assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
    "Ratios must sum to 1.0"

  labels_array = np.array(labels)
  np.random.shuffle(labels_array)

  n_total = len(labels_array)
  n_train = int(n_total * train_ratio)
  n_val = int(n_total * val_ratio)

  train_labels = labels_array[:n_train]
  val_labels = labels_array[n_train:n_train + n_val]
  test_labels = labels_array[n_train + n_val:]

  return train_labels, val_labels, test_labels


def save_splits_to_files(train_labels, val_labels, test_labels, output_dir):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  train_file = output_dir / "train_labels.txt"
  with open(train_file, 'w', encoding='utf-8') as f:
    for label in sorted(train_labels):
      f.write(f"{label}\n")

  val_file = output_dir / "val_labels.txt"
  with open(val_file, 'w', encoding='utf-8') as f:
    for label in sorted(val_labels):
      f.write(f"{label}\n")

  test_file = output_dir / "test_labels.txt"
  with open(test_file, 'w', encoding='utf-8') as f:
    for label in sorted(test_labels):
      f.write(f"{label}\n")


def process(dataset_dir, dataset_name, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
  dataset_dir = Path(dataset_dir)

  # Extract the actual dataset filename from dataset_name
  dataset_filename = dataset_name.split('/')[-1]
  csv_file = dataset_dir / f"{dataset_filename}.csv"

  if not csv_file.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

  unique_labels = get_unique_labels(csv_file)

  train_labels, val_labels, test_labels = split_labels(
    unique_labels,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio
  )

  save_splits_to_files(train_labels, val_labels, test_labels, dataset_dir)


def parse_args():
  parser = argparse.ArgumentParser(description="Split TMNIST dataset")
  parser.add_argument('--dataset_name', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True, help='Directory to save dataset')
  parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
  return parser.parse_args()


def main(args):
  np.random.seed(args.seed)

  output_dir = Path(args.output_dir)

  dataset_dir = download_kaggle_dataset(output_dir, args.dataset_name)
  process(dataset_dir, args.dataset_name)


if __name__ == "__main__":
  args = parse_args()
  main(args)

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os


def download_kaggle_dataset(output_dir, dataset_name):
  # Create a subdirectory for this specific dataset
  dataset_filename = dataset_name.split('/')[-1]
  dataset_dir = Path(output_dir) / dataset_filename
  dataset_dir.mkdir(parents=True, exist_ok=True)

  # Check if dataset already exists by looking for any CSV file
  csv_files = list(dataset_dir.glob("*.csv"))
  if csv_files:
    print(f"Dataset already exists at {dataset_dir}")
    return dataset_dir

  try:
    # Check for environment variables before importing kaggle
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
      raise RuntimeError(
        "Kaggle credentials not found. Please set environment variables:\n"
        "  KAGGLE_USERNAME: your Kaggle username\n"
        "  KAGGLE_KEY: your Kaggle API key\n\n"
        "In Google Colab, use:\n"
        "  from google.colab import userdata\n"
        "  os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\n"
        "  os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')\n\n"
        "Or alternatively, create kaggle.json in ~/.kaggle/\n"
        "See: https://github.com/Kaggle/kaggle-api#api-credentials"
      )

    import kaggle
    print(f"Downloading dataset {dataset_name}...")
    kaggle.api.dataset_download_files(
      dataset_name,
      path=str(dataset_dir),
      unzip=True
    )
    print(f"Dataset downloaded to {dataset_dir}")
    return dataset_dir
  except ImportError:
    raise ImportError("Kaggle API not installed. Install with: pip install kaggle")
  except Exception as e:
    raise RuntimeError(f"Failed to download dataset: {e}")


def get_unique_labels(csv_path):
  df = pd.read_csv(csv_path)

  # Detect which column name is used for labels
  if 'label' in df.columns:
    label_column = 'label'
  elif 'labels' in df.columns:
    label_column = 'labels'
  else:
    raise ValueError(f"Neither 'label' nor 'labels' column found in dataset. Available columns: {df.columns.tolist()}")

  labels = df[label_column].unique()
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

  # Find any CSV file in the dataset directory
  csv_files = list(dataset_dir.glob("*.csv"))
  if not csv_files:
    raise FileNotFoundError(f"No CSV file found in dataset directory: {dataset_dir}")

  # Use the first found CSV file
  csv_file = csv_files[0]

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

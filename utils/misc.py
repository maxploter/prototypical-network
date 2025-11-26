"""
Miscellaneous utility functions for the prototypical network project.
"""

import random

import numpy as np
import torch


def is_thresholded_dataset(dataset_name, dataset_path):
  # For TMNIST, check if the path contains 'thresholded'
  if dataset_name == 'tmnist':
    dataset_path_str = str(dataset_path).lower()
    return 'thresholded' in dataset_path_str

  # Chess dataset uses discrete piece values (0-12), treated as multi-class classification
  # We'll use CrossEntropyLoss and multi-class metrics
  if dataset_name == 'chess':
    return True

  # Default to False for unknown datasets
  return False


def set_seed(seed):
  """
  Set random seeds for reproducibility across all libraries.

  Args:
      seed (int): Random seed value
  """
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For completely deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

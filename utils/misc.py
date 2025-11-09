"""
Miscellaneous utility functions for the prototypical network project.
"""


def is_thresholded_dataset(dataset_name, dataset_path):
  # For TMNIST, check if the path contains 'thresholded'
  if dataset_name == 'tmnist':
    dataset_path_str = str(dataset_path).lower()
    return 'thresholded' in dataset_path_str

  # Default to False for unknown datasets
  return False

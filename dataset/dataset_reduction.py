"""
Dataset reduction strategies for controlling training dataset size.

This module provides a flexible, extensible framework for reducing dataset size
using different strategies:
1. Percentage reduction: Simply reduce the dataset to a percentage of its original size
2. Class variability reduction: Remove a percentage of classes (future implementation)
3. Per-class sample reduction: Reduce samples per class (future implementation)
"""

import numpy as np
import torch
from torch.utils.data import Subset


class DatasetReductionStrategy:
  """Base class for dataset reduction strategies."""

  def __init__(self, reduction_factor):
    """
    Args:
        reduction_factor: Float between 0 and 1 indicating how much data to keep.
                         E.g., 0.5 means keep 50% of the data.
    """
    if not 0 < reduction_factor <= 1.0:
      raise ValueError(f"reduction_factor must be between 0 and 1, got {reduction_factor}")
    self.reduction_factor = reduction_factor

  def apply(self, dataset):
    """
    Apply the reduction strategy to the dataset.

    Args:
        dataset: The dataset to reduce

    Returns:
        Reduced dataset (usually a Subset)
    """
    raise NotImplementedError("Subclasses must implement apply()")


class PercentageReductionStrategy(DatasetReductionStrategy):
  """
  Randomly sample a percentage of the dataset.

  This strategy keeps the class distribution but reduces the overall size.
  """

  def __init__(self, reduction_factor):
    """
    Args:
        reduction_factor: Float between 0 and 1 indicating percentage to keep
    """
    super().__init__(reduction_factor)

  def apply(self, dataset):
    """
    Randomly sample indices to create a reduced dataset.

    Args:
        dataset: The dataset to reduce

    Returns:
        Subset of the original dataset
    """
    total_size = len(dataset)
    reduced_size = int(total_size * self.reduction_factor)

    indices = np.random.choice(total_size, size=reduced_size, replace=False)
    indices = sorted(indices.tolist())

    print(f"PercentageReductionStrategy: Reduced dataset from {total_size} to {reduced_size} samples "
          f"({self.reduction_factor * 100:.1f}%)")

    return Subset(dataset, indices)


class ClassVariabilityReductionStrategy(DatasetReductionStrategy):
  """
  Remove a percentage of classes from the dataset.

  This strategy reduces class variability while keeping all samples
  from the remaining classes.

  Future implementation for experimenting with reduced class diversity.
  """

  def __init__(self, reduction_factor):
    """
    Args:
        reduction_factor: Float between 0 and 1 indicating percentage of classes to keep
    """
    super().__init__(reduction_factor)

  def apply(self, dataset):
    """
    Remove random classes from the dataset.

    Args:
        dataset: The dataset to reduce (must have 'targets' attribute)

    Returns:
        Subset of the original dataset with reduced classes
    """
    # Get targets - handle both direct datasets and wrapped datasets
    if hasattr(dataset, 'targets'):
      targets = dataset.targets
    elif hasattr(dataset, 'base_dataset') and hasattr(dataset.base_dataset, 'targets'):
      targets = dataset.base_dataset.targets
    else:
      raise ValueError("Dataset must have 'targets' attribute or be wrapped with base_dataset")

    # Convert to numpy if tensor
    if isinstance(targets, torch.Tensor):
      targets = targets.numpy()

    # Get unique classes
    unique_classes = np.unique(targets)
    total_classes = len(unique_classes)

    # Calculate how many classes to keep
    num_classes_to_keep = max(1, int(total_classes * self.reduction_factor))

    # Use global random state (controlled by np.random.seed() in main)
    classes_to_keep = np.random.choice(unique_classes, size=num_classes_to_keep, replace=False)
    classes_to_keep_set = set(classes_to_keep)

    # Get indices of samples belonging to selected classes
    indices = [i for i, target in enumerate(targets) if target in classes_to_keep_set]

    print(f"ClassVariabilityReductionStrategy: Reduced from {total_classes} to {num_classes_to_keep} classes "
          f"({self.reduction_factor * 100:.1f}%), keeping {len(indices)} samples")

    return Subset(dataset, indices)


class StratifiedReductionStrategy(DatasetReductionStrategy):
  """
  Reduce dataset while maintaining class distribution.

  This strategy samples the same percentage from each class,
  ensuring the class balance is preserved.

  Future implementation for balanced reduction.
  """

  def __init__(self, reduction_factor):
    """
    Args:
        reduction_factor: Float between 0 and 1 indicating percentage to keep from each class
    """
    super().__init__(reduction_factor)

  def apply(self, dataset):
    """
    Sample from each class proportionally.

    Args:
        dataset: The dataset to reduce (must have 'targets' attribute)

    Returns:
        Subset of the original dataset with preserved class distribution
    """
    # Get targets
    if hasattr(dataset, 'targets'):
      targets = dataset.targets
    elif hasattr(dataset, 'base_dataset') and hasattr(dataset.base_dataset, 'targets'):
      targets = dataset.base_dataset.targets
    else:
      raise ValueError("Dataset must have 'targets' attribute or be wrapped with base_dataset")

    # Convert to numpy if tensor
    if isinstance(targets, torch.Tensor):
      targets = targets.numpy()

    # Get unique classes
    unique_classes = np.unique(targets)

    # Collect indices for each class
    selected_indices = []
    for class_label in unique_classes:
      class_indices = np.where(targets == class_label)[0]
      num_samples = len(class_indices)
      num_to_keep = max(1, int(num_samples * self.reduction_factor))

      # Use global random state (controlled by np.random.seed() in main)
      sampled = np.random.choice(class_indices, size=num_to_keep, replace=False)
      selected_indices.extend(sampled.tolist())

    # Sort indices for consistency
    selected_indices = sorted(selected_indices)

    print(f"StratifiedReductionStrategy: Reduced dataset from {len(targets)} to {len(selected_indices)} samples "
          f"({self.reduction_factor * 100:.1f}%), preserving class distribution across {len(unique_classes)} classes")

    return Subset(dataset, selected_indices)


def build_reduction_strategy(strategy_name, reduction_factor):
  """
  Factory function to build a dataset reduction strategy.

  Args:
      strategy_name: Name of the strategy ('percentage', 'class_variability', 'stratified')
      reduction_factor: Float between 0 and 1 indicating how much data to keep

  Returns:
      DatasetReductionStrategy instance
  """
  strategies = {
    'percentage': PercentageReductionStrategy,
    'class_variability': ClassVariabilityReductionStrategy,
    'stratified': StratifiedReductionStrategy,
  }

  if strategy_name not in strategies:
    raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

  return strategies[strategy_name](reduction_factor)


def apply_dataset_reduction(dataset, reduction_factor=1.0, strategy='percentage'):
  """
  Apply dataset reduction if needed.

  Args:
      dataset: The dataset to potentially reduce
      reduction_factor: Float between 0 and 1. If 1.0, no reduction is applied.
      strategy: Name of the reduction strategy to use

  Returns:
      Either the original dataset (if reduction_factor == 1.0) or a reduced Subset

  Note:
      Uses global random state - ensure np.random.seed() is set before calling
  """
  if reduction_factor == 1.0:
    # No reduction needed
    return dataset

  if reduction_factor <= 0:
    raise ValueError(f"reduction_factor must be positive, got {reduction_factor}")

  # Build and apply the strategy
  reduction_strategy = build_reduction_strategy(strategy, reduction_factor)
  reduced_dataset = reduction_strategy.apply(dataset)

  return reduced_dataset

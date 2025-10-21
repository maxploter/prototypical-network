import torch
from torch import Tensor
from torch.utils.data import BatchSampler
from typing import Iterator, List


class EpisodeSampler(BatchSampler):

  def __init__(
    self,
    labels: Tensor,
    classes_per_iteration: int,
    samples_per_class: int,
    number_of_iterations: int
  ):
    self.labels = labels
    self.classes_per_iteration = classes_per_iteration
    self.samples_per_class = samples_per_class
    self.number_of_iterations = number_of_iterations

    self.unique_labels, self.label_counts = torch.unique(self.labels, return_counts=True)
    max_count = torch.max(self.label_counts).item()
    self.indeces_per_class = torch.zeros((self.unique_labels.size(0), max_count), dtype=torch.long)

    for i, label in enumerate(self.unique_labels):
      label_indices = torch.nonzero(self.labels == label).squeeze()
      self.indeces_per_class[i, :len(label_indices)] = label_indices

  def __len__(self) -> int:
    return self.number_of_iterations

  def __iter__(self) -> Iterator[List[int]]:
    for _ in range(self.number_of_iterations):
      episode_indices = []
      selected_classes = torch.randperm(self.unique_labels.size(0))[:self.classes_per_iteration]

      for class_idx in selected_classes:
        class_indices = self.indeces_per_class[class_idx]
        class_counts = self.label_counts[class_idx]
        indices = torch.randperm(class_counts)[:self.samples_per_class]
        sampled_indices = class_indices[indices]
        episode_indices.extend(sampled_indices.tolist())

      yield episode_indices

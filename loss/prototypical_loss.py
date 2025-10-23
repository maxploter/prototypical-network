import torch
import torch.nn as nn


class PrototypicalLoss(nn.Module):
    """
    Loss function for Prototypical Networks
    """
    def __init__(self, k_shot):
      super(PrototypicalLoss, self).__init__()
      self.k_shot = k_shot

    def forward(self, embeddings, targets):

      targets = targets.to('cpu')
      embeddings = embeddings.to('cpu')

      classes = torch.unique(targets)
      n_way = len(classes)
      n_query = torch.sum(targets == classes[0]).item() - self.k_shot

      prototypes = torch.stack([
          chunk[:self.k_shot].mean(0)
          for chunk in torch.split(embeddings, embeddings.size(0) // n_way)
      ])

      query_samples = torch.cat([
          chunk[self.k_shot:]
          for chunk in torch.split(embeddings, embeddings.size(0) // n_way)
      ])

      # Get query labels
      query_labels = torch.cat([
          torch.full((n_query,), i, dtype=torch.long)
          for i in range(n_way)
      ])

      # Compute Euclidean distances from each query to each prototype
      # Distance: ||query - prototype||^2
      # Shape: (n_query_total, n_way)
      dists = torch.cdist(query_samples, prototypes, p=2.0) ** 2

      # Compute log probabilities using negative distances
      log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)

      # Compute loss as negative log likelihood
      loss = torch.nn.functional.nll_loss(log_p_y, query_labels)

      return loss

import torch
import torch.nn as nn


class PrototypicalLoss(nn.Module):
    """
    Loss function for Prototypical Networks
    """
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, support_embeddings, query_embeddings, n_way, k_shot, q_query):
        """
        Compute the prototypical loss

        Args:
            support_embeddings: Embeddings of support samples
            query_embeddings: Embeddings of query samples
            n_way: Number of classes per episode
            k_shot: Number of support samples per class
            q_query: Number of query samples per class

        Returns:
            loss: The computed loss
        """
        # TODO: Implement the actual loss computation
        return torch.tensor(0.0, requires_grad=True)

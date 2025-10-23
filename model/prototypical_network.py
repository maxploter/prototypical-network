import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalCnnNetwork(nn.Module):
    """
    Prototypical Network for few-shot learning.
    Uses a CNN encoder to embed images into a learned metric space.
    """

    def __init__(self, embedding_dim=64):
        super(PrototypicalCnnNetwork, self).__init__()

        # CNN encoder for MNIST (28x28 grayscale images)
        self.encoder = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            # Conv block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7

            # Conv block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 -> 3x3

            # Conv block 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 3x3 -> 1x1
        )

        # Final embedding layer
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            embeddings: Embedded representations of shape (batch_size, embedding_dim)
        """
        batch_size = x.size(0)

        # Pass through encoder
        x = self.encoder(x)
        x = x.view(batch_size, -1)

        # Pass through final embedding layer
        embeddings = self.fc(x)

        return embeddings

class PrototypicalAutoencoder(nn.Module):

  def __init__(self, encoder, encoder_dim=64):
    super(PrototypicalAutoencoder, self).__init__()
    self.encoder = encoder

    self.prototypical_network = nn.Sequential(
      nn.Linear(encoder_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, 2),
    )

  def forward(self, x):
    x = self.encoder(x)
    embeddings = self.prototypical_network(x)
    return embeddings
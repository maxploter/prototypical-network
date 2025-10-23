import torch.nn as nn

class Autoencoder(nn.Module):
  def __init__(self, encoding_dim=32):
    super(Autoencoder, self).__init__()

    # Encoder
    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, encoding_dim),
      # nn.ReLU(),
    )

    # Decoder
    self.decoder = nn.Sequential(
      nn.Linear(encoding_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 28 * 28),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = x.view(x.size(0), -1)
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
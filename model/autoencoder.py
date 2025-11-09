import torch.nn as nn

class Autoencoder(nn.Module):
  def __init__(self, encoding_dim=32, return_logits=False):
    """
    Args:
      encoding_dim: Dimension of the encoded representation
      return_logits: If True, output raw logits (use with BCEWithLogitsLoss for binary data)
                     If False, apply sigmoid to decoder output (use with MSELoss for regression)
    """
    super(Autoencoder, self).__init__()
    self.return_logits = return_logits

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
    )

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    # Apply sigmoid activation unless we want raw logits
    if not self.return_logits:
      decoded = self.sigmoid(decoded)

    return decoded
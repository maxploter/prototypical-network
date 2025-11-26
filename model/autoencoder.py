import torch.nn as nn

class Autoencoder(nn.Module):
  def __init__(self, encoding_dim=32, return_logits=False, num_classes=None):
    """
    Simple autoencoder for reconstruction or classification.

    For Chess (multi-class classification):
      - Input: (B, 64) with piece indices 0-12
      - Output: (B, 13, 64) with logits for 13 classes at each of 64 positions
      - The 13 dimension is required by CrossEntropyLoss to compute softmax

    Args:
      encoding_dim: Dimension of the encoded representation
      return_logits: If True, output raw logits (use with BCEWithLogitsLoss for binary data or CrossEntropyLoss for multi-class)
                     If False, apply sigmoid to decoder output (use with MSELoss for regression)
      num_classes: If specified, output multi-class logits with this many classes per position (for chess: 13)
                   Output shape will be (B, num_classes, num_positions) for CrossEntropyLoss
                   Output shape will be (B, num_classes, num_positions) for CrossEntropyLoss
    """
    super(Autoencoder, self).__init__()
    self.return_logits = return_logits
    self.num_classes = num_classes

    # Determine output dimension
    # For chess with 64 squares and 13 classes: 64 * 13 = 832
    # For chess with 64 squares and 13 classes: decoder outputs 64 * 13 = 832 values
    input_dim = 28 * 28  # Will be inferred from input
    output_dim = input_dim if num_classes is None else input_dim * num_classes

    # Encoder
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, 128),
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
      nn.Linear(128, output_dim),
    )

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    batch_size = x.shape[0]
    num_positions = x.shape[1]

    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    # For multi-class output, reshape to (B, C, num_positions) for CrossEntropyLoss
    # CrossEntropyLoss needs class dimension at position 1
    if self.num_classes is not None:
      decoded = decoded.view(batch_size, num_positions, self.num_classes)
      # Reshape to (B, num_positions, num_classes) then transpose
      decoded = decoded.transpose(1, 2)  # (B, C, num_positions)
      # Transpose to (B, num_classes, num_positions)
      # This gives us logits for each class at each position
      decoded = decoded.transpose(1, 2)  # (B, 13, 64) for chess
      # Return raw logits (no softmax/activation - CrossEntropyLoss applies softmax internally)
    # For binary/regression output: (B, num_positions)
    # Apply sigmoid activation unless we want raw logits
    if not self.return_logits:
      decoded = self.sigmoid(decoded)

    return decoded
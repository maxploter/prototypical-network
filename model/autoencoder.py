import torch.nn as nn

class Autoencoder(nn.Module):
  def __init__(self, encoding_dim=32, return_logits=False, num_classes=None, input_dim=784):
    """
    Simple autoencoder for reconstruction or classification.

    For Chess (multi-class classification per position):
      - Input: (B, 64) with piece indices 0-12
      - Output: (B, 13, 64) with logits for 13 classes at each of 64 positions
      - CrossEntropyLoss expects shape (B, C, N) where C is num_classes

    For MNIST (binary reconstruction):
      - Input: (B, 784) with pixel values 0-1
      - Output: (B, 784) with reconstructed pixel values

    Args:
      encoding_dim: Dimension of the encoded representation
      return_logits: If True, output raw logits (use with BCEWithLogitsLoss or CrossEntropyLoss)
                     If False, apply sigmoid to decoder output (use with MSELoss)
      num_classes: If specified, output multi-class logits with this many classes per position
                   Output shape will be (B, num_classes, num_positions) for CrossEntropyLoss
      input_dim: Input dimension (64 for chess, 784 for MNIST)
    """
    super(Autoencoder, self).__init__()
    self.return_logits = return_logits
    self.num_classes = num_classes
    self.input_dim = input_dim

    # Determine output dimension
    # For multi-class: output logits for each class at each position
    # e.g., chess: 64 positions * 13 classes = 832 output values
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
    num_positions = x.shape[1]  # 64 for chess, 784 for MNIST

    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    # For multi-class output, reshape to (B, num_classes, num_positions)
    if self.num_classes is not None:
      # decoded is (B, num_positions * num_classes)
      # Reshape to (B, num_positions, num_classes) then transpose to (B, num_classes, num_positions)
      decoded = decoded.view(batch_size, num_positions, self.num_classes)
      decoded = decoded.transpose(1, 2)  # (B, num_classes, num_positions)
      # Return raw logits - CrossEntropyLoss applies softmax internally
      return decoded

    # For binary/regression output: (B, num_positions)
    # Apply sigmoid activation unless we want raw logits
    if not self.return_logits:
      decoded = self.sigmoid(decoded)

    return decoded
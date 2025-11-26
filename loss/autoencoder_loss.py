import torch.nn as nn


class AutoencoderLoss(nn.Module):
  """
  Loss function for Autoencoder supporting MSE, BCE, and CrossEntropy
  """

  def __init__(self, loss_type='mse'):
    """
    Args:
        loss_type: 'mse' for regression (use with sigmoid output)
                   'bce' for binary classification (use with logits)
                   'ce' for multi-class classification (use with logits)
    """
    super(AutoencoderLoss, self).__init__()

    self.loss_type = loss_type

    if loss_type == 'bce':
      # BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
      # Use this when the model outputs raw logits (return_logits=True)
      self.loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'ce':
      # CrossEntropyLoss for multi-class classification
      # Expects: outputs (B, C, d1, ...) with logits, targets (B, d1, ...) with class indices
      # For chess: outputs (B, 13, 64), targets (B, 64)
      # The 13 dimension contains logits for each of the 13 possible pieces at each square
      # CrossEntropyLoss internally applies softmax over the 13 logits and computes NLL loss
      self.loss_fn = nn.CrossEntropyLoss()
    else:
      # MSE for regression
      # Use this when the model outputs sigmoid values (return_logits=False)
      self.loss_fn = nn.MSELoss()

    # Define which losses are used for backprop
    self.weight_dict = {f'loss_{loss_type}': 1.0}

  def forward(self, outputs, targets):
    """
    Compute loss

    Args:
        outputs: Model outputs
                 - For BCE/MSE: (B, num_positions) with logits or sigmoid values
                 - For CE: (B, C, num_positions) with C classes, logits for each class
                   Example for chess: (B, 13, 64) - 13 logits for each of 64 squares
        targets: Target values
                 - For BCE/MSE: (B, num_positions) pixel values
                 - For CE: (B, num_positions) class indices (values 0 to C-1)
                   Example for chess: (B, 64) with values 0-12 indicating piece at each square

    Returns:
        dict: Dictionary with losses
    """
    loss = self.loss_fn(outputs, targets)

    # Return dict of losses
    return {
      f'loss_{self.loss_type}': loss,
    }

import torch.nn as nn


class AutoencoderLoss(nn.Module):
  """
  Loss function for Autoencoder supporting both MSE and BCE
  """

  def __init__(self, loss_type='mse'):
    """
    Args:
        loss_type: 'mse' for regression (use with sigmoid output) or 'bce' for binary classification (use with logits)
    """
    super(AutoencoderLoss, self).__init__()

    self.loss_type = loss_type

    if loss_type == 'bce':
      # BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
      # Use this when the model outputs raw logits (return_logits=True)
      self.loss_fn = nn.BCEWithLogitsLoss()
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
        outputs: Model outputs (logits if loss_type='bce', sigmoid values if loss_type='mse')
        targets: Target pixels

    Returns:
        dict: Dictionary with losses
    """
    loss = self.loss_fn(outputs, targets)

    # Return dict of losses
    return {
      f'loss_{self.loss_type}': loss,
    }

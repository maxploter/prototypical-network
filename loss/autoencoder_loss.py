import torch
import torch.nn as nn


class AutoencoderLoss(nn.Module):
    """
    Loss function for Autoencoder
    """
    def __init__(self):
        super(AutoencoderLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # Define which losses are used for backprop
        self.weight_dict = {'loss_mse': 1.0}

    def forward(self, outputs, targets):
        """
        Compute MSE loss

        Returns:
            dict: Dictionary with losses
        """
        mse = self.mse_loss(outputs, targets)

        # Return dict of losses
        return {
            'loss_mse': mse,    # Used for backprop (in weight_dict)
        }

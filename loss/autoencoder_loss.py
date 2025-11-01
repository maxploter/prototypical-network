import torch
import torch.nn as nn
import math


class AutoencoderLoss(nn.Module):
    """
    Loss function for Autoencoder with PSNR metric
    """
    def __init__(self):
        super(AutoencoderLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # Define which losses are used for backprop
        self.weight_dict = {'loss_mse': 1.0}

    def forward(self, outputs, targets):
        """
        Compute MSE loss and PSNR metric

        Returns:
            dict: Dictionary with losses and metrics
        """
        mse = self.mse_loss(outputs, targets)

        # Compute PSNR (for logging, not backprop)
        mse_value = mse.item()
        if mse_value == 0:
            psnr = torch.tensor(float('inf'))
        else:
            psnr = torch.tensor(10 * math.log10(1.0 / mse_value))

        # Return dict of losses
        return {
            'loss_mse': mse,    # Used for backprop (in weight_dict)
            'psnr': psnr        # Not in weight_dict, so just logged
        }

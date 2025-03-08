import torch
from torch import nn

class MeanAbsoluteDirectionalLoss(nn.Module):
    """
    Loss function that calculates the mean absolute directional loss between the predicted and the expected values.
    """
    def __init__(self):
        super(MeanAbsoluteDirectionalLoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Parameters
        ----------
        y_pred : Predicted values
        y_true : Expected values

        Returns
        -------
        Loss value
        """
        # Calculate the directional loss
        adl = -1 * torch.sign(y_pred * y_true) * torch.abs(y_true)
        return adl.mean()
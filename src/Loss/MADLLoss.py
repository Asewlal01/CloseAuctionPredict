import torch
from torch import nn

class MADLLoss(nn.Module):
    """
    Loss function that calculates the mean absolute directional loss between the predicted and the expected values.
    """
    def __init__(self):
        """
        Initialize the Mean Absolute Directional Loss object.
        """
        super(MADLLoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mean Absolute Directional Loss.

        Parameters
        ----------
        y_pred : Predicted values
        y_true : Expected values

        Returns
        -------
        Loss value
        """
        # Calculate the directional loss
        adl = torch.sign(y_pred * y_true) * torch.abs(y_true)
        return -adl.mean()
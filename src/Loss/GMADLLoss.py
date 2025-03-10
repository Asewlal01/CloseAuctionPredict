import torch
from numpy.ma.core import negative
from torch import nn

class GMADLLoss(nn.Module):
    """
    Loss function that calculates the general mean absolute directional loss between the predicted and the expected
    values. This loss function is different from the MADLoss, as it is always differentiable.
    """
    def __init__(self, a, b):
        """
        Initialize the Mean Absolute Directional Loss object.

        Parameters
        ----------
        a : Slope of GMADL around 0
        b : Parameter that increases the reward of higher returns
        """
        super(GMADLLoss, self).__init__()

        self.a = a
        self.b = b

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
        negative_term = -1
        exponential_term = 1 + torch.exp(-self.a * y_true * y_pred)
        return_term = torch.abs(y_true) ** self.b

        adl = negative_term * (1 / exponential_term - 0.5) * return_term

        return adl.mean()
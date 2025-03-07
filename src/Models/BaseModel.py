import torch.nn as nn
import torch
from torch.utils import data

class BaseModel(nn.Module):
    """
    Abstract class that represents a model. It should be inherited by all the Models in the project as it provides
    training functionality and other useful methods that are common to all the Models.
    """

    def __init__(self, expected_dim: int):
        """
        Initializes the BaseModel class. It should be called by all the classes that inherit from this class.

        Parameters
        ----------
        expected_dim : Expected number of dimensions after unsqueezing the input tensor.
        """
        super(BaseModel, self).__init__()
        self.layers = [Unsqueeze(expected_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model. This method should be implemented by all the Models that inherit from
        this class as it is the main method that is called when the model is used.

        Parameters
        ----------
        x : Input tensor

        Returns
        -------
        Output tensor

        """
        for layer in self.layers:
            x = layer(x)
        return x

class Unsqueeze(nn.Module):
    """
    Layer designed to unsqueeze a tensor. This is needed when the input is not given as a batch, which may cause
    errors in the forward pass.
    """

    def __init__(self, expected_dim: int):
        super(Unsqueeze, self).__init__()
        self.expected_dim = expected_dim

    def forward(self, x):
        # If only one sample is given, add a batch dimension
        if x.dim() == self.expected_dim - 1:
            return x.unsqueeze(0)
        return x
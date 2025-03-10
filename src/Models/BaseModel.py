import torch.nn as nn
import torch

class BaseModel(nn.Module):
    """
    Abstract class that represents a model. It should be inherited by all the Models in the project as it provides
    training functionality and other useful methods that are common to all the Models.
    """

    def __init__(self, expected_dim: int, dropout: float = 0.5):
        """
        Initializes the BaseModel class. It should be called by all the classes that inherit from this class.

        Parameters
        ----------
        expected_dim : Expected number of dimensions after unsqueezing the input tensor.
        dropout : Dropout rate to use in the fully connected layers.
        """
        super(BaseModel, self).__init__()
        self.layers = [Unsqueeze(expected_dim)]
        self.output_dim = None
        self.dropout = dropout
        self.build_model()
        self.output_layer()


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

    def build_model(self) -> None:
        """
        Build the model by adding all layers to self.layers. This method should be implemented by all the Models
        that inherit from this class.
        """
        raise NotImplementedError("Subclasses must implement `build_model`")

    def output_layer(self):
        """
        Adds the output layer to the model. This method requires self.output_dim to be defined, as it is the dimension
        of the layer before the output layer.
        """
        if self.output_dim is None:
            raise AttributeError("The attribute `output_dim` is not defined in the model.")

        self.layers.append(
            nn.Dropout(self.dropout)
        )

        self.layers.append(
            nn.Linear(self.output_dim, 1)
        )
        self.layers = nn.ModuleList(self.layers)


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
import torch.nn as nn
import torch
from Layers.Unsqueeze import Unsqueeze

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
        self.device = 'cpu'
        self.layers: list[nn.Module] = [Unsqueeze(expected_dim)]
        self.output_dim: int = 0
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

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model, and apply sigmoid activation function to the output.

        Parameters
        ----------
        x : Input tensor

        Returns
        -------
        Output tensor

        """
        x = self.forward(x)
        return torch.sigmoid(x)

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
        if self.output_dim == 0:
            raise AttributeError("The attribute `output_dim` is not defined in the model.")

        self.layers.append(
            nn.Linear(self.output_dim, 1)
        )

        self.layers = nn.ModuleList(self.layers)

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.
        """
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self
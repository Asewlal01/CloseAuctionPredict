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
        super().__init__()
        self.device = 'cpu'
        # All layers that are not fully connected are stored here
        self.layers: list[nn.Module] = [Unsqueeze(expected_dim)]

        # Fully connected layers
        self.fc_layers: list[nn.Module] = []
        self.fc_neurons: list[int] = []

        self.output_dim: int = 0
        self.build_model()
        self.build_fully_connected_layers()


    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model. This method should be implemented by all the Models that inherit from
        this class as it is the main method that is called when the model is used.

        Parameters
        ----------
        x : Input tensor
        z: Input tensor with information of yesterday's closing return and overnight return

        Returns
        -------
        Output tensor

        """
        for layer in self.layers:
            x = layer(x)

        # Combining x and z into a single tensor
        # x = torch.cat((x, z), dim=1)
        for layer in self.fc_layers:
            x = layer(x)

        return x

    def predict(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model, and apply sigmoid activation function to the output.

        Parameters
        ----------
        x : Input tensor
        z: Input tensor with information of yesterday's closing return and overnight return

        Returns
        -------
        Output tensor

        """
        x = self.forward(x, z)
        return torch.sigmoid(x)

    def build_model(self) -> None:
        """
        Build the model by adding all layers to self.layers. This method should be implemented by all the Models
        that inherit from this class.
        """
        raise NotImplementedError("Subclasses must implement `build_model`")

    def build_fully_connected_layers(self):
        """
        Add the fully connected layers at the end of the model. This method requires self.fc_neurons to be defined.
        Returns
        """
        if self.output_dim == 0:
            raise AttributeError("The attribute `output_dim` is not defined in the model.")

        # Add two to the input dimension because we are concatenating the input tensor with z
        # self.output_dim = self.output_dim + 2
        self.output_dim = self.output_dim
        if self.fc_neurons is None:
            return

        input_dim = self.output_dim
        for out_neurons in self.fc_neurons:
            self.fc_layers.append(
                nn.Linear(input_dim, out_neurons)
            )
            input_dim = out_neurons

        # Output layer
        self.fc_layers.append(
            nn.Linear(input_dim, 1)
        )

        self.layers = nn.ModuleList(self.layers)
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.
        """
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self

    def reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
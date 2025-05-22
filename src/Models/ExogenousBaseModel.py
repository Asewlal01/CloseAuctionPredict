import torch
from Models.BaseModel import BaseModel

class ExogenousBaseModel(BaseModel):
    """
    Abstract class that represents a model. It should be inherited by all the Models in the project as it provides
    training functionality and other useful methods that are common to all the Models.
    """

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model. This method should be implemented by all the Models that inherit from
        this class as it is the main method that is called when the model is used.

        Parameters
        ----------
        x : Input tensor
        z : Exogenous tensor

        Returns
        -------
        Output tensor

        """
        # Model specific layers
        for layer in self.layers:
            x = layer(x)

        # Concatenate the exogenous features with the output of the model
        x = torch.cat((x, z), dim=1)

        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        return x

    def predict(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model, and apply sigmoid activation function to the output.

        Parameters
        ----------
        x : Input tensor

        Returns
        -------
        Output tensor

        """
        x = self.forward(x, z)
        return torch.sigmoid(x)

    def build_fully_connected_layers(self):
        """
        Add the fully connected layers at the end of the model. This method requires self.fc_neurons to be defined.
        Returns
        """
        if self.output_dim == 0:
            raise AttributeError("The attribute `output_dim` is not defined in the model.")

        # 2 Exogenous features
        self.output_dim = self.output_dim + 2
        super().build_fully_connected_layers()
from Models.BaseModel import BaseModel
from torch import nn

class BaseLogisticRegression(BaseModel):
    """
    Base class for all the Logistic Regression Models. It inherits from the BaseModel class.
    """
    def __init__(self, feature_size: int, sequence_size: int):
        """
        Initializes the Logistic Regression Model.

        Parameters
        ----------
        feature_size : Number of features in each time step
        sequence_size : Number of time steps in the input data
        """

        self.feature_size = feature_size
        self.sequence_size = sequence_size

        expected_dims = 3
        super().__init__(expected_dims, 0)

    def build_model(self) -> None:
        """
        Build the model by adding all layers to self.layers.
        """

        self.layers.append(ViewLayer())

        # Each time step has feature_size features, hence total features is feature_size * sequence_size
        feature_size = self.feature_size * self.sequence_size
        self.output_dim = feature_size

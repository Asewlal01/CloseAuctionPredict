from Models.BaseModel import BaseModel
from torch import nn

class BaseLogisticRegression(BaseModel):
    def __init__(self, feature_size: int, sequence_size: int):
        expected_dims = 2
        super().__init__(expected_dims)

        self.layers.append(nn.Linear(feature_size * sequence_size, 1))

        # Save the layers
        self.layers = nn.ModuleList(self.layers)


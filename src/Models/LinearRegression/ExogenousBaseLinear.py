from Models.ExogenousBaseModel import ExogenousBaseModel
from Models.LinearRegression.BaseLinear import BaseLinear

class ExogenousBaseLinear(ExogenousBaseModel, BaseLinear):
    def __init__(self, feature_size: int, sequence_size: int):


        self.feature_size = feature_size
        self.sequence_size = sequence_size

        expected_dims = 3
        ExogenousBaseModel.__init__(self, expected_dims)

    def build_model(self):
        BaseLinear.build_model(self)
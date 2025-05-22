from Models.LinearRegression.ExogenousBaseLinear import ExogenousBaseLinear

class ExogenousLobLinear(ExogenousBaseLinear):
    """
    Logistic Regression model for predicting Stock Prices using Limit Order Book data.
    """
    def __init__(self, sequence_size: int):
        """
        Initializes the Logistic Regression Model.

        Parameters
        ----------
        sequence_size : Number of time steps in the input data
        """
        feature_size = 20
        super().__init__(feature_size, sequence_size)

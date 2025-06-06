from Models.LinearRegression.ExogenousBaseLinear import ExogenousBaseLinear

class ExogenousTradeLobLinear(ExogenousBaseLinear):
    """
    Logistic Regression model for predicting Stock Prices using Limit Order Book data and Trade data.
    """
    def __init__(self, sequence_size: int):
        """
        Initializes the Logistic Regression Model.

        Parameters
        ----------
        sequence_size : Number of time steps in the input data
        """
        feature_size = 29
        super().__init__(feature_size, sequence_size)

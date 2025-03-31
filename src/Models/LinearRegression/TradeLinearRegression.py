from Models.LinearRegression.BaseLinearRegression import BaseLinearRegression

class TradeLinearRegression(BaseLinearRegression):
    """
    Logistic Regression model for predicting the direction of the stock price.
    """
    def __init__(self, sequence_size: int):
        """
        Initializes the Logistic Regression Model.

        Parameters
        ----------
        sequence_size : Number of time steps in the input data
        """
        # Trade data has 5 features: Open, High, Low, Close, Volume
        feature_size = 2
        super().__init__(feature_size, sequence_size)

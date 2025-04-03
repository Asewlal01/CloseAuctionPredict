from Models.LinearRegression.BaseLinearRegression import BaseLinearRegression

class SimpleLinearRegression(BaseLinearRegression):
    def __init__(self):
        """
        Initializes the Simple Linear Regression Model.
        """
        # Simple Linear Regression has 2 features only: Return and Volume
        feature_size = 2
        sequence_size = 1
        super().__init__(feature_size, sequence_size)
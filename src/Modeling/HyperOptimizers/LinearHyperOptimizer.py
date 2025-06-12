from Models.LinearRegression.LobLinear import LobLinear
from Modeling.HyperOptimizers.BaseHyperOptimizer import BaseHyperOptimizer
from optuna import Trial

class LinearHyperOptimizer(BaseHyperOptimizer):
    """
    Class to optimize the hyperparameters of the Linear Regression model using Optuna.
    """
    def generate_model(self, trial: Trial, sequence_size: int) -> LobLinear:
        """
        Generate a Linear Regression model with hyperparameters sampled from the trial.

        Parameters
        ----------
        trial : Optuna trial object containing the hyperparameters.
        sequence_size : Sequence size for the model input.

        Returns
        -------
        Linear model with randomly generated hyperparameters.
        """
        model = LobLinear(sequence_size)

        return model
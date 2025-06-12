from Models.ReccurentModels.LobLSTM import LobLSTM
from Modeling.HyperOptimizers.BaseHyperOptimizer import BaseHyperOptimizer
from optuna import Trial
import torch

class RecurrentHyperOptimizer(BaseHyperOptimizer):
    """
    Class to optimize the hyperparameters of the Recurrent model using Optuna.
    """
    def generate_model(self, trial: Trial, sequence_size: int) -> LobLSTM:
        """
        Generate a Recurrent model with hyperparameters sampled from the trial.

        Parameters
        ----------
        trial : Optuna trial object containing the hyperparameters.
        sequence_size : Sequence size for the model input.

        Returns
        -------
        Recurrent model with randomly generated hyperparameters.
        """
        hidden_size_min, hidden_size_max, hidden_size_step = self.config['lstm']['hidden_size']
        lstm_size_min, lstm_size_max, lstm_size_step = self.config['lstm']['lstm_size']

        hidden_size = trial.suggest_int('hidden_size', hidden_size_min, hidden_size_max, step=hidden_size_step)
        lstm_size = trial.suggest_int('lstm_size', lstm_size_min, lstm_size_max, step=lstm_size_step)
        fc_neurons, dropout = self.generate_common_parameter(trial)

        model = LobLSTM(hidden_size, lstm_size, fc_neurons, dropout=dropout)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        return model
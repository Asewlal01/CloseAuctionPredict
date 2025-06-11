from Models.TransformerModels.ExogenousTradeLobTransformer import ExogenousTradeLobTransformer
from Modeling.HyperOptimizers.BaseHyperOptimizer import BaseHyperOptimizer
from optuna import Trial
import torch

class TransformerHyperOptimizer(BaseHyperOptimizer):
    """
    Class to optimize the hyperparameters of the Transformer model using Optuna.
    """
    def generate_model(self, trial: Trial, sequence_size: int) -> ExogenousTradeLobTransformer:
        """
        Generate a Transformer model with hyperparameters sampled from the trial.

        Parameters
        ----------
        trial : Optuna trial object containing the hyperparameters.
        sequence_size : Sequence size for the model input.

        Returns
        -------
        Transformer model with randomly generated hyperparameters.
        """

        # Sample hyperparameters for the Transformer model
        transformer_config = self.config['transformer']
        embedding_size_min, embedding_size_max, embedding_size_step = transformer_config['embedding_size']
        num_heads_min, num_heads_max, num_heads_step = transformer_config['num_heads']
        dim_feedforward_min, dim_feedforward_max, dim_feedforward_step = transformer_config['dim_feedforward']
        num_layers_min, num_layers_max, num_layers_step = transformer_config['num_layers']

        embedding_size = trial.suggest_int('embedding_size', embedding_size_min, embedding_size_max, step=embedding_size_step)
        num_heads = trial.suggest_int('num_heads', num_heads_min, num_heads_max, step=num_heads_step)
        dim_feedforward = trial.suggest_int('dim_feedforward', dim_feedforward_min, dim_feedforward_max, step=dim_feedforward_step)
        num_layers = trial.suggest_int('num_layers', num_layers_min, num_layers_max, step=num_layers_step)

        fc_neurons, dropout = self.generate_common_parameter(trial)

        model = ExogenousTradeLobTransformer(
            sequence_size=sequence_size,
            embedding_size=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            fc_neurons=fc_neurons
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        return model
from Models.ConvolutionalModels.ExogenousTradeLobConvolve import ExogenousTradeLobConvolve
from Modeling.HyperOptimizers.BaseHyperOptimizer import BaseHyperOptimizer
from optuna import Trial
import torch

class ConvolutionalHyperOptimizer(BaseHyperOptimizer):
    """
    Class to optimize the hyperparameters of the Convolutional model using Optuna.
    """
    def generate_model(self, trial: Trial, sequence_size: int) -> ExogenousTradeLobConvolve:
        """
        Generate a Convolutional model with hyperparameters sampled from the trial.

        Parameters
        ----------
        trial : Optuna trial object containing the hyperparameters.
        sequence_size : Sequence size for the model input.

        Returns
        -------
        Convolutional model with randomly generated hyperparameters.
        """

        conv_layers_min, conv_layers_max, conv_layers_step = self.config['cnn']['conv_layers']
        channels_min, channels_max, channels_step = self.config['cnn']['channels']
        kernel_min, kernel_max, kernel_step = self.config['cnn']['kernel_size']

        conv_layers = trial.suggest_int('conv_channels', conv_layers_min, conv_layers_max, step=conv_layers_step)

        conv_channels = []
        kernel_sizes = []
        for layer in range(conv_layers):
            channels = trial.suggest_int(f'conv_channels_{layer}', channels_min, channels_max, step=channels_step)
            kernel_size = trial.suggest_int(f'kernel_size_{layer}', kernel_min, kernel_max, step=kernel_step)

            conv_channels.append(channels)
            kernel_sizes.append(kernel_size)

        fc_neurons, dropout = self.generate_common_parameter(trial)

        model = ExogenousTradeLobConvolve(sequence_size, conv_channels, fc_neurons, kernel_sizes, dropout=dropout)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        return model


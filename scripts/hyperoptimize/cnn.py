from Models.ConvolutionalModels.LobConvolve import LobConvolve
from Modeling.HyperOptimizer import HyperOptimizer
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager
import os
import multiprocessing

def model_parameters(trial, sequence_size):
    # Number of layers
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 5)
    n_fc_layers = trial.suggest_int('n_fc_layers', 1, 5)

    # Convolutional layer parameters
    conv_channels = []
    kernel_sizes = []
    for i in range(n_conv_layers):
        # Every convolutional channel is a multiple of 8
        conv_channels_multiple = trial.suggest_int(f'conv_channels_multiple_{i}', 4, 64)
        conv_channels.append(conv_channels_multiple * 8)

        # Kernel size is an uneven number with a minimum of 3 (2n + 1)
        kernel_sizes_multiple = trial.suggest_int(f'kernel_sizes_multiple_{i}', 1, 4)
        kernel_sizes.append(kernel_sizes_multiple * 2 + 1)

    # Fully connected layer parameters
    fc_neurons = []
    for i in range(n_fc_layers):
        # Every neuron in the layer is a multiple of 8
        fc_neurons_multiple = trial.suggest_int(f'fc_neurons_multiple_{i}', 1, 64)
        fc_neurons.append(fc_neurons_multiple * 8)

    # Dropout rate are multiple of 0.1
    dropout_rate_multiple = trial.suggest_int('dropout_power', 0, 5)
    dropout_rate = dropout_rate_multiple * 0.1

    model = LobConvolve(
        sequence_size=sequence_size,
        conv_channels=conv_channels,
        fc_neurons=fc_neurons,
        kernel_size=kernel_sizes,
        dropout=dropout_rate
    )
    model.to('cuda')

    return model, sequence_size

def objective(trial):
    # Sequence size is a multiple of 8
    sequence_size_multiple = trial.suggest_int('sequence_size_multiple', 7, 45)
    sequence_size = sequence_size_multiple * 8

    # Model parameters
    model, sequence_size = model_parameters(trial, sequence_size)

    # Learning rate is a power of 10
    lr_power = trial.suggest_int('lr_power', -4, -2)
    lr = 10 ** lr_power

    # Number of epochs is multiple of 5
    epochs_multiple = trial.suggest_int('epochs_multiple', 1, 6)
    epochs = epochs_multiple * 5

    return model, epochs, lr, sequence_size

def run_study():
    path_to_data = 'data/intraday'

    # Save path of results
    results = 'results/hyperparameters_intraday'
    name = 'cnn'

    # NUmber of evaluations
    trials = 100

    # Save path of results
    os.makedirs(results, exist_ok=True)

    # Train the model
    train_manager = BaseDatasetManager(path_to_data, 1)
    test_manager = BaseDatasetManager(path_to_data, 1)
    optimizer = HyperOptimizer(train_manager, test_manager, '2021-1', 11)
    optimizer.optimize(objective, name, results, n_trials=trials)

if __name__ == '__main__':
    run_study()



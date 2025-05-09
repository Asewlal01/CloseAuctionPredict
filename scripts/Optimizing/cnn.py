from Models.ConvolutionalModels.LimitOrderBookConvolve import LimitOrderBookConvolve
from Modeling.HyperOptimizer import HyperOptimizer
from Modeling.DatasetManager import DatasetManager
import os

def objective(trial):
    sequence_size = 420

    # Number of layers
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 5)
    n_fc_layers = trial.suggest_int('n_fc_layers', 1, 5)

    # Convolutional layer parameters
    conv_channels = []
    kernel_sizes = []
    for i in range(n_conv_layers):
        # Every convolutional channel is a multiple of 16
        conv_channels_multiple = trial.suggest_int(f'conv_channels_multiple_{i}', 1, 8)
        conv_channels.append(conv_channels_multiple * 8)

        # Kernel size is an uneven number with a minimum of 3 (2n + 1)
        kernel_sizes_multiple = trial.suggest_int(f'kernel_sizes_multiple_{i}', 1, 4)
        kernel_sizes.append(kernel_sizes_multiple * 2 + 1)

    # Fully connected layer parameters
    fc_neurons = []
    for i in range(n_fc_layers):
        # Every neuron in the layer is a multiple of 16
        fc_neurons_multiple = trial.suggest_int(f'fc_neurons_multiple_{i}', 1, 8)
        fc_neurons.append(fc_neurons_multiple * 8)

    # Dropout rate are multiple of 0.1
    dropout_rate_multiple = trial.suggest_int('dropout_rate_multiple', 0, 5)
    dropout_rate = dropout_rate_multiple * 0.1

    # Learning rate is a power of 10
    lr_power = trial.suggest_int('lr_power', -5, -2)
    lr = 10 ** lr_power

    # Number of epochs is multiple of 5
    epochs_multiple = trial.suggest_int('epochs_multiple', 1, 10)
    epochs = epochs_multiple * 5

    # Setup model
    model = LimitOrderBookConvolve(
        sequence_size=sequence_size,
        conv_channels=conv_channels,
        fc_neurons=fc_neurons,
        kernel_size=kernel_sizes,
        dropout=dropout_rate
    )

    return model, epochs, lr

if __name__ == '__main__':
    path_to_data = 'data/merged_files'

    # Save path of results
    results = 'results/hyperparameters'
    os.makedirs(results, exist_ok=True)

    # Train the model
    dataset = DatasetManager(path_to_data)
    optimizer = HyperOptimizer(dataset, '2021-1')
    save_path = os.path.join(results, 'cnn')
    optimizer.optimize(objective, results, n_trials=100)



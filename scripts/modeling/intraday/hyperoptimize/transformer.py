from Models.TransformerModels.LobTransformer import LobTransformer
from Modeling.HyperOptimizer import HyperOptimizer
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager
import os

def model_parameters(trial, sequence_size):

    # Embedding size is multiple of 16
    embedding_size_multiple = trial.suggest_int('embedding_size_multiple', 1, 4)
    embedding_size = 16 * embedding_size_multiple

    # Number of heads is power of 2
    num_heads_power = trial.suggest_int('num_heads_power', 0, 2)
    num_heads = 2 ** num_heads_power

    # Feed forward size is a multiple of 16
    dim_feedforward_multiple = trial.suggest_int('dim_feedforward_multiple', 4, 8)
    dim_feedforward = dim_feedforward_multiple * embedding_size

    # Number of layers is a multiple of 1
    num_layers_multiple = trial.suggest_int('num_layers_multiple', 1, 2)
    num_layers = 2 * num_layers_multiple

    # Number of layers
    n_fc_layers = trial.suggest_int('n_fc_layers', 1, 5)

    # Fully connected layer parameters
    fc_neurons = []
    for i in range(n_fc_layers):
        # Every neuron in the layer is a multiple of 8
        fc_neurons_multiple = trial.suggest_int(f'fc_neurons_multiple_{i}', 1, 8)
        fc_neurons.append(fc_neurons_multiple * 8)

    # Dropout rate are multiple of 0.1
    dropout_rate_multiple = trial.suggest_int('dropout_power', 0, 5)
    dropout_rate = dropout_rate_multiple * 0.1

    model = LobTransformer(
        sequence_size=sequence_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        fc_neurons = fc_neurons,
        dropout = dropout_rate,
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
    name = 'transformer'

    # NUmber of evaluations
    trials = 100

    # Save path of results
    os.makedirs(results, exist_ok=True)

    # Train the model
    train_manager = IntradayDatasetManager(path_to_data, 1)
    test_manager = IntradayDatasetManager(path_to_data, 1)
    optimizer = HyperOptimizer(train_manager, test_manager, '2021-1', 11)
    optimizer.optimize(objective, name, results, n_trials=trials)

if __name__ == '__main__':
    run_study()
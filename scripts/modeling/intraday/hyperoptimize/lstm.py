from Models.ReccurentModels.LobLSTM import LobLSTM
from Modeling.HyperOptimizer import HyperOptimizer
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager
import os

def model_parameters(trial, sequence_size):
    # Hidden size is multiple of 8
    hidden_size_multiple = trial.suggest_int(f'hidden_size_multiple', 1, 8)
    hidden_size = 8 * hidden_size_multiple

    # LSTM layers
    lstm_size = trial.suggest_int(f'lstm_size', 1, 4)

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

    model = LobLSTM(
        hidden_size=hidden_size,
        lstm_size = lstm_size,
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
    name = 'lstm'

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



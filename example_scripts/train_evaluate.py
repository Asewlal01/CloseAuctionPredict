from Modeling.ModelRunner import ModelRunner
from Modeling.DatasetManagers.ClosingDatasetManager import ClosingDatasetManager
from Models.ReccurentModels.ExogenousTradeLobLSTM import ExogenousTradeLobLSTM
import torch

# Load the dataset manager
path_to_data = 'example_data/dataset'
dataset_manager = ClosingDatasetManager(path_to_data)

# Setting up the dataset
training_days = 30  # Number of days to use for training
validation_days = 5 # Number of days to use for validation
test_days = 5       # Number of days to use for testing
dataset_manager.setup_dataset(training_days, validation_days, test_days)

# Initialize the model
model = ExogenousTradeLobLSTM(hidden_size=16, lstm_size=2, fc_neurons=[64, 32], dropout=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set up the model runner
sequence_size = 60      # Input sequence size for the model
use_trading = True      # Whether to use trading features
use_exogenous = True    # Whether to use exogenous features
tester = ModelRunner(model, dataset_manager, sequence_size, use_trading, use_exogenous)

# Training phase
epochs = 10             # Number of epochs to train the model
learning_rate = 1e-3    # Learning rate for the optimizer
stopping_epochs = 10    # Controls early stopping
verbose = True # Print training progress
tester.train(epochs, learning_rate, stopping_epochs, verbose)

# Get predictions for evaluation month
predictions = tester.predictions_on_test()

# Used to compute accuracy
match_count = 0
sample_count = 0

# Predictions returns a list of tensors, each tensor corresponds to a day
for prediction in predictions:
    # First column are predicted values, second column are actual values
    y_pred, y_true = prediction[:, 0], prediction[:, 1]

    # Count the number of sign matches
    match_count += ((y_pred > 0) == (y_true > 0)).sum().item()
    sample_count += len(y_pred)
print(f"Accuracy: {match_count / sample_count * 100:.2f}%")


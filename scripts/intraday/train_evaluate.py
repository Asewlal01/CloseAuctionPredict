from Modeling.WalkForwardTester import WalkForwardTester
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager
from Models.ReccurentModels.LobLSTM import LobLSTM
import torch

# Load the dataset manager
path_to_data = 'data/dataset/intraday'
train_manager = IntradayDatasetManager(path_to_data, 3) # 3 Months of training data
test_manager = IntradayDatasetManager(path_to_data, 1) # 1 Month of testing data

train_starting_month = '2022-1'
validation_starting_month = '2022-4'
train_manager.setup_dataset(train_starting_month)
test_manager.setup_dataset(validation_starting_month)

# Initialize the model
model = LobLSTM(hidden_size=16, lstm_size=2, fc_neurons=[64, 32], dropout=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create the walk forward tester
sequence_size = 100     # Number of sequence steps in the input data
use_trading = False     # Not included with intraday data
use_exogenous = False   # Not included with intraday data
tester = WalkForwardTester(model, train_manager, test_manager, sequence_size, use_trading, use_exogenous)

# Training phase
epochs = 20
learning_rate = 1e-3
verbose = True # Print training progress
tester.train(epochs, learning_rate, verbose)

# Get predictions for evaluation month
predictions = tester.test_predictions()

# Predictions are given for each day
match_count = 0
sample_count = 0
for prediction in predictions:
    # First column are predicted values, second column are actual values
    y_pred, y_true = prediction[:, 0], prediction[:, 1]

    # Count the number of sign matches
    match_count += ((y_pred > 0) == (y_true > 0)).sum().item()
    sample_count += len(y_pred)

print(f"Accuracy: {match_count / sample_count * 100:.2f}%")


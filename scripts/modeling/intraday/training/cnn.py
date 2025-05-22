from Models.ConvolutionalModels.LobConvolve import LobConvolve
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager
import os
from Modeling.WalkForwardTester import WalkForwardTester
import torch
from tqdm import tqdm

def get_model():
    """
    Get the model to be used for training.
    """
    # Create the model
    model = LobConvolve(
        sequence_size=120,
        conv_channels=[128, 64],
        fc_neurons=[128, 64],
        kernel_size=[3, 5],
        dropout=0.1,
    )
    model.to('cuda')

    return model

def run_training(model, epochs, lr, sequence_size, name):
    """
    Run the training of the model.
    """
    # Path to data
    path_to_data = 'data/intraday'

    # Save path of results
    results = f'results/intraday/params/{name}'

    # Create the directory if it does not exist
    os.makedirs(results, exist_ok=True)

    # Create the dataset manager
    print('Loading Data')
    train_manager = IntradayDatasetManager(path_to_data, 4)
    train_manager.setup_dataset('2021-9')
    print('Data Loaded')

    # Create tester
    tester = WalkForwardTester(model, train_manager, train_manager, sequence_size=sequence_size)

    MONTHS_TO_LOOP = 12
    for i in tqdm(range(MONTHS_TO_LOOP)):
        tester.train(epochs, lr, False, True, 0.1)
        print(f'Training completed for month {i + 1} of {MONTHS_TO_LOOP}')

        # Save the model
        model_path = os.path.join(results, f'model_{i}.pt')
        torch.save(model.state_dict(), model_path)

        if i + 1 < MONTHS_TO_LOOP:
            # Increment the dataset
            train_manager.increment_dataset()
            model.reset_parameters()

if __name__ == '__main__':
    model = get_model()
    sequence_size = 120
    epochs = 100
    lr = 1e-4
    name = 'cnn'
    run_training(model, epochs, lr, sequence_size, name)


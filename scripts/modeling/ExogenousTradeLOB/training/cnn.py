from Models.ConvolutionalModels.ExogenousTradeLobConvolve import ExogenousTradeLobConvolve
from Modeling.DatasetManagers.ClosingDatasetManager import ClosingDatasetManager
import os
from Modeling.WalkForwardTester import WalkForwardTester
import torch
from tqdm import tqdm

PATH_TO_DATA = 'data/closing'

def get_model():
    """
    Get the model to be used for training.
    """
    # Create the model
    model = ExogenousTradeLobConvolve(
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

    # Save path of results
    results = f'results/ExogenousTradeLOB/params/{name}'

    # Create the directory if it does not exist
    os.makedirs(results, exist_ok=True)

    # Create the dataset manager
    print('Loading Data')
    train_manager = ClosingDatasetManager(PATH_TO_DATA, 12)
    train_manager.setup_dataset('2021-1')
    print('Loaded Data')

    # Create tester
    tester = WalkForwardTester(model, train_manager, train_manager, sequence_size=sequence_size, use_trading=True, use_exogenous=True)

    MONTHS_TO_LOOP = 12
    for i in tqdm(range(MONTHS_TO_LOOP)):
        tester.train(epochs, lr, False, True, 1/12)
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


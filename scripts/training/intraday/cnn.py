from Models.ConvolutionalModels.LobConvolve import LobConvolve
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager
import os
from Modeling.WalkForwardTester import WalkForwardTester
import torch

def run_training():
    """
    Run the training of the model.
    """
    # Path to data
    path_to_data = 'data/intraday'

    # Save path of results
    results = 'results/intraday_params/cnn'

    # Create the directory if it does not exist
    os.makedirs(results, exist_ok=True)

    # Create the dataset manager
    train_manager = BaseDatasetManager(path_to_data, 3)
    train_manager.setup_dataset('2021-10')

    # Create the model
    model = LobConvolve(
        sequence_size=64,
        conv_channels=[12*8, 57*8],
        fc_neurons=[13*8, 59*8],
        kernel_size=[3, 7],
        dropout=0.4,
    )
    model.to('cuda')
    epochs = 30
    lr = 1e-3

    # Create tester
    tester = WalkForwardTester(model, train_manager, train_manager, sequence_size=64)

    MONTHS_TO_LOOP = 12
    for i in range(MONTHS_TO_LOOP):
        tester.train(epochs, lr, 1, verbose=False)
        print(f'Training completed for month {i + 1} of {MONTHS_TO_LOOP}')

        # Save the model
        model_path = os.path.join(results, f'model_{i}.pt')
        torch.save(model.state_dict(), model_path)

        if i + 1 < MONTHS_TO_LOOP:
            # Increment the dataset
            train_manager.increment_dataset()
            model.reset_parameters()

if __name__ == '__main__':
    run_training()


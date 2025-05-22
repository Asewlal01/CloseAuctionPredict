import torch
import os
from typing import Tuple
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager

date_type = list[int]

ExogenousDatasetTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class ClosingDatasetManager(IntradayDatasetManager):
    def __init__(self, dataset_path: str, num_months: int):
        """
        Initialize the dataset manager.

        Parameters
        ----------
        dataset_path : Path to the dataset directory. Assumed to contain subfolders which represent the data for each month
        num_months : Number of months to load
        """
        super().__init__(dataset_path, num_months)

        # Different types of data since it also has exogenous features
        self.dataset: list[list[ExogenousDatasetTuple]] = []

    def load_dataset(self, month_paths: list[str]) -> None:
        """
        Load the dataset from the specified directory.

        Parameters
        ----------
        month_paths : List of paths to the months to load.

        """
        if len(month_paths) == 0:
            raise ValueError('No months paths provided.')

        # Load data for each month
        for month_path in month_paths:
            data = load_data(month_path)
            self.dataset.append(data)


def load_data(month_path: str) -> list[ExogenousDatasetTuple]:
    """
    Load the data for a specific month. It is assumed that the folder of the month contains subfolders for each
    day within that month. It is assumed that the subfolder contains X, y and z files, which can all be
    loaded into torch tensors as .pt files. The X file is assumed to be the input variable, y is the output variable
    and z is the additional time-independent features.

    Parameters
    ----------
    month_path : Month for which to load the data. The month is assumed to be in the format 'YYYY-MM'.

    Returns
    -------
    Tuple of torch tensors as (X, y, z).
    """
    # Sorting to make sure the data is in the correct order
    days = os.listdir(month_path)
    days.sort()

    results = []
    for day in days:
        # Path to each file
        path_X = os.path.join(month_path, day, 'x.pt')
        path_y = os.path.join(month_path, day, 'y.pt')
        path_Z = os.path.join(month_path, day, 'z.pt')

        # Load the data
        X = torch.load(path_X, weights_only=False)
        y = torch.load(path_y, weights_only=False)
        z = torch.load(path_Z, weights_only=False)

        # Make sure y is two-dimensional
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        results.append((X, y, z))

    return results
import torch
import os
from typing import Tuple
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager

# Represents a tuple containing the data for a single intraday dataset sample in the format (X, y).
IntradaySampleTuple = Tuple[torch.Tensor, torch.Tensor]
class IntradayDatasetManager(BaseDatasetManager):
    """
    Dataset manager for the intraday dataset.
    """
    def load_single_day(self, date_path: str) -> IntradaySampleTuple:
        """
        Load the intraday data for a specific day.


        Parameters
        ----------
        date_path : Path to the directory containing the data for a specific day.
        The day is assumed to be in the format 'YYYY-MM-DD'.

        Returns
        -------
        Tuple of tensors as (X, y).
        """
        return load_data(date_path)


def load_data(date_path: str) -> IntradaySampleTuple:
    """
    Load the data for a specific month. It is assumed that the folder of the month contains subfolders for each
    day within that month. It is assumed that the subfolder contains X, y and z files, which can all be
    loaded into torch tensors as .pt files. The X file is assumed to be the input variable, y is the output variable
    and z is the additional time-independent features.

    Parameters
    ----------
    date_path : Month for which to load the data. The month is assumed to be in the format 'YYYY-MM'.

    Returns
    -------
    Tuple of torch tensors as (X, y, z).
    """
    # Path to each file
    path_X = os.path.join(date_path, 'x.pt')
    path_y = os.path.join(date_path, 'y.pt')

    # Load the data
    X = torch.load(path_X, weights_only=False)
    y = torch.load(path_y, weights_only=False)

    return X, y
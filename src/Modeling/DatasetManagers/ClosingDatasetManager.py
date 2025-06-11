import torch
import os
from typing import Tuple
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager
from Modeling.DatasetManagers.IntradayDatasetManager import load_data as load_intraday_data

# Represents a tuple containing the data for a single intraday dataset sample in the format (X, y, z).
ClosingSampleTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
class ClosingDatasetManager(BaseDatasetManager):
    """
    Dataset manager for the closing dataset.
    """
    def load_single_day(self, date_path: str) -> ClosingSampleTuple:
        """
        Load the closing data for a specific day.

        Parameters
        ----------
        date_path : Path to the directory containing the data for a specific day.
        The day is assumed to be in the format 'YYYY-MM-DD'.

        Returns
        -------
        Tuple of tensors as (X, y, z).
        """
        return load_data(date_path)

def load_data(date_path: str) -> ClosingSampleTuple:
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
    # Intraday also has X and y, so we reuse the load_intraday_data function
    X, y = load_intraday_data(date_path)

    # Loading the exogenous features z
    path_z = os.path.join(date_path, 'z.pt')
    z = torch.load(path_z, weights_only=False)

    return X, y, z
import torch
import pandas as pd
import numpy as np
import os
from typing import Tuple

DatasetTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]



class DatasetManager:
    def __init__(self, dataset_path: str, normalise: bool=True, horizon: int=0, sequence_size: int=440,
                 whole_sequence: bool=False):
        """
        Initialize the dataset manager.

        Parameters
        ----------
        dataset_path : Path to the dataset directory. Assumed to contain subfolders which represent the data for each
        month
        normalise : Boolean indicating whether to z-normalise the data.
        horizon : Number of time steps to predict.
        sequence_size : Number of sequence steps in the input data.
        whole_sequence : Boolean indicating whether to load the whole sequence or a single value.
        """
        self.dataset_path = dataset_path
        self.normalise = normalise
        self.horizon = horizon
        self.sequence_size = sequence_size
        self.whole_sequence = whole_sequence

        self.train_month_start = None
        self.train_month_end = None
        self.validation_month = None
        self.test_month = None

        self.train = None
        self.validation = None
        self.test = None


    def setup_dataset(self, start_month: str, end_month: str):
        """
        Set up the dataset by loading data from the specified directory.

        Parameters
        ----------
        start_month : Starting month for the training data.
        end_month : Ending month for the training data.
        """
        self.train_month_start = split_month(start_month)
        self.train_month_end = split_month(end_month)
        year_start, month_start = self.train_month_start
        year_end, month_end = self.train_month_end

        for year in range(year_start, year_end + 1):
            # If start and end are different, we need to go through all the months of a year
            end_range = month_end if year == year_end else 12

            for month in range(month_start, end_range + 1):
                month_path = os.path.join(self.dataset_path, f'{year}-{month}')
                if not os.path.exists(month_path):
                    raise FileNotFoundError(f'Directory {month_path} does not exist.')

                # Load the data for the month
                X, y, z = load_data(month_path, whole_sequence=self.whole_sequence, horizon=self.horizon, sequence_size=self.sequence_size)
                # Check if nan in X
                if torch.isnan(X).any():
                    raise ValueError(f'NaN values found in X for month {month_path}')

                # If the train set is empty, we need to set it up
                if self.train is None:
                    self.train = (X, y, z)

                # Otherwise we can simply combine the datasets
                else:
                    self.train = combine_dataset(self.train, (X, y, z))

        # Validation and test are in the next year
        if month_end == 12:
            validation_month = f'{year_end + 1}-1'
            test_month = f'{year_end + 1}-2'
        # Test set is in the next year
        elif month_end == 11:
            validation_month = f'{year_end}-12'
            test_month = f'{year_end + 1}-1'
        # All of them are in the same year
        else:
            validation_month = f'{year_end}-{month_end + 1}'
            test_month = f'{year_end}-{month_end + 2}'

        validation_path = os.path.join(self.dataset_path, validation_month)
        if not os.path.exists(validation_path):
            raise FileNotFoundError(f'Directory {validation_path} does not exist.')
        test_path = os.path.join(self.dataset_path, test_month)
        if not os.path.exists(test_path):
            raise FileNotFoundError(f'Directory {test_path} does not exist.')

        self.validation_month = validation_month
        self.test_month = test_month

        self.validation = load_data(validation_path, whole_sequence=self.whole_sequence, horizon=self.horizon, sequence_size=self.sequence_size)
        self.test = load_data(test_path, whole_sequence=self.whole_sequence, horizon=self.horizon, sequence_size=self.sequence_size)

    def increment_dataset(self):
        """
        Increment the dataset by adding one month. The train set will be a combination of the previous train and validation sets,
        and the validation set will be the previous test set. The test set will be the next month.

        Returns
        -------
        None
        """

        if self.train_month_start is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        # Recompute the train month start and end
        start_year, start_month = self.train_month_start
        end_year, end_month = self.train_month_end

        # New year and month
        start_month += 1
        if start_month > 12:
            start_year += 1
            start_month = 1

        end_month += 1
        if end_month > 12:
            end_year += 1
            end_month = 1

        self.setup_dataset(f'{start_year}-{start_month}', f'{end_year}-{end_month}')

    def get_training_data(self, binary_classification: bool=True) -> tuple[DatasetTuple, DatasetTuple]:
        """
        Get the training and validation data. The y values of the training dataset are conve

        Parameters
        ----------
        binary_classification : Boolean indicating whether the data will be used for training. If True, the output of the
        train set will be converted to a binary classification problem.

        Returns
        -------
        Tuple containing the training and validation sets.
        """
        if self.train is None or self.validation is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        train, validation = self.train, self.validation

        if self.normalise:
            train, validation = normalize_datasets(self.train, self.validation)

        if binary_classification:
            X, y, z = train
            y = convert_to_classification(y)
            train = (X, y, z)

        return train, validation

    def get_test_data(self) -> tuple[DatasetTuple, DatasetTuple, DatasetTuple]:
        """
        Get the data for the train, validation and test sets.

        Returns
        -------
        Tuple containing the train, validation and test sets.
        """
        if self.train is None or self.validation is None or self.test is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        if not self.normalise:
            return self.train, self.validation, self.test

        return normalize_datasets(self.train, self.validation, self.test)

def load_data(month: str, whole_sequence: bool, horizon: int, sequence_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the data for a specific month. It is assumed that closing_returns are the variable to be predicted and that the
    current_returns, overnight_returns, previous_returns and relative_traded_volume are the features. Current_returns
    and relative_traded_volume can return the whole sequence.


    Parameters
    ----------
    month : Month for which to load the data.
    whole_sequence : Boolean indicating whether to load the whole sequence or a single value.
    horizon : Number of time steps to predict.
    sequence_size : Number of sequence steps in the input data.

    Returns
    -------
    Tuple containing the input and target variables.
    """

    # Depending on the horizon and whether to load the whole sequence or a single value, we need to select the columns
    single_columns = [f'{sequence_size - horizon - 1}']
    sequence_columns = [f'{i}' for i in range(sequence_size - horizon)]
    sequence_columns = sequence_columns if whole_sequence else single_columns

    current_returns = pd.read_parquet(f'{month}/current_returns.parquet', columns=sequence_columns)
    relative_traded_volume = pd.read_parquet(f'{month}/relative_traded_volume.parquet', columns=sequence_columns)
    if whole_sequence:
        X = np.stack([current_returns, relative_traded_volume], axis=2)
    else:
        X = np.hstack([current_returns, relative_traded_volume])
    X = torch.tensor(X, dtype=torch.float)

    overnight_returns = pd.read_parquet(f'{month}/overnight_returns.parquet')
    previous_returns = pd.read_parquet(f'{month}/previous_returns.parquet', columns=single_columns)
    z = np.hstack([overnight_returns, previous_returns])
    z = torch.tensor(z, dtype=torch.float)

    y = pd.read_parquet(f'{month}/closing_returns.parquet', columns=single_columns).values
    y = torch.tensor(y, dtype=torch.float)

    return X, y, z

def split_month(month: str) -> list[int]:
    """
    Split the month into a year-month format. For example, '2021-01' becomes 2021 and 1.
    Parameters
    ----------
    month : Month to split.

    Returns
    -------
    Year and month as integers.
    """
    year, month = month.split('-')
    return [int(year), int(month)]

def combine_dataset(set_1, set_2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine the two datasets. It is assumed that the datasets are in format: X, y, z. The datasets are concatenated
    along the first dimension.


    Parameters
    ----------
    set_1 : First dataset to combine.
    set_2 : Second dataset to combine.

    Returns
    -------
    Combined dataset as a tensor.
    """
    X1, y1, z1 = set_1
    X2, y2, z2 = set_2
    X = torch.cat([X1, X2], dim=0)
    y = torch.cat([y1, y2], dim=0)
    z = torch.cat([z1, z2], dim=0)

    return X, y, z

def normalize_datasets(train, *datasets):
    """
    Normalize the datasets using the mean and standard deviation of the training set. The mean and the standard
    deviation from the training set are used to normalize the other datasets.
    The datasets are assumed to be in the format: X, y, z, with only X and z being normalized.

    Parameters
    ----------
    train : Training dataset to use for normalization.
    datasets : Other datasets to normalize.

    Returns
    -------
    Normalized datasets.
    """
    # Unpacking allows for easier access to the datasets
    X_train, y_train, z_train = train

    # Statistics for z-normalisation
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True)
    z_mean = z_train.mean(dim=0, keepdim=True)
    z_std = z_train.std(dim=0, keepdim=True)

    X_train = (X_train - X_mean) / X_std
    z_train = (z_train - z_mean) / z_std
    yield X_train, y_train, z_train

    # Normalising the other datasets
    for dataset in datasets:
        X, y, z = dataset
        X = (X - X_mean) / X_std
        z = (z - z_mean) / z_std

        yield X, y, z

def convert_to_classification(y):
    """
    Convert the output of the dataset to a binary classification problem. The output is 1 if the value is greater than 0
    and 0 otherwise. This is used for binary classification problems.
    Parameters
    ----------
    y : Output of the dataset to convert.

    Returns
    -------
    Binary classification output.
    """
    return torch.where(y > 0, 1, 0).float()




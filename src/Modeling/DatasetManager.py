import torch
import os
from typing import Tuple
import gc
from DataProcessing.DatasetAssembler import generate_dates, date_type

DatasetTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class DatasetManager:
    def __init__(self, dataset_path: str, train_length: int=9):
        """
        Initialize the dataset manager.

        Parameters
        ----------
        dataset_path : Path to the dataset directory. Assumed to contain subfolders which represent the data for each
        month
        train_length : Length of training set in months. Default is 10 months.
        """
        self.dataset_path = dataset_path
        self.train_length = train_length

        # Currently test length is set to 1 month
        self.test_length = 1
        self.train_start = None
        self.train_end = None
        self.test_month = None

        self.train = None
        self.test = None

    def setup_dataset(self, start_month: str):
        """
        Set up the dataset by loading data from the specified directory.

        Parameters
        ----------
        start_month : Starting month for the training data.
        """
        self.train_start = split_month(start_month)
        year_start, month_start = self.train_start

        # End of training set
        # Subtracted by 1 since first month is included in the training set
        year_end, month_end = increment_month(year_start, month_start, self.train_length - 1)
        self.train_end = [year_end, month_end]

        # Generate all the months to loop over
        months = generate_dates(self.train_start, self.train_end)
        month_paths = [generate_month_path(self.dataset_path, month) for month in months]

        # Load all files
        results = [load_data(month) for month in month_paths]
        self.train = combine_dataset(*results)

        # Test is one month after the training set
        test_year, test_month = increment_month(year_end, month_end, self.test_length)

        # Converting to string format
        test_month = f'{test_year}-{test_month}'

        test_path = os.path.join(self.dataset_path, test_month)
        if not os.path.exists(test_path):
            raise FileNotFoundError(f'Directory {test_path} does not exist.')

        self.test_month = test_month
        self.test = load_data(test_path)

    def increment_dataset(self):
        """
        Increment the dataset by adding one month. The train set will be a combination of the previous train and validation sets,
        and the validation set will be the previous test set. The test set will be the next month.
        """

        # Completely clear the memory
        self.empty_dataset()

        if self.train_start is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        # Recompute the train month start and end
        start_year, start_month = self.train_start
        incremented_year, incremented_month = increment_month(start_year, start_month, 1)

        self.setup_dataset(f'{incremented_year}-{incremented_month}')

    def empty_dataset(self):
        """
        Empty the dataset. This is used to clear the memory after the dataset has been used.
        """
        del self.train
        del self.test
        self.train = None
        self.test = None
        torch.cuda.empty_cache()
        gc.collect()

    def get_training_data(self, binary_classification: bool=True) -> DatasetTuple:
        """
        Get the training and validation data. The y values of the training dataset are converted to a binary classification

        Parameters
        ----------
        binary_classification : Boolean indicating whether the data will be used for training. If True, the output of the
        train set will be converted to a binary classification problem.

        Returns
        -------
        Tuple containing the training and validation sets.
        """
        if self.train is None:
            raise ValueError('Training data not set up. Please call setup_dataset() first.')

        train = self.train
        # Normalizing the datasets
        if binary_classification:
            X, y, z = train
            y = convert_to_classification(y)
            train = (X, y, z)

        return train

    def get_test_data(self) -> tuple[DatasetTuple, DatasetTuple]:
        """
        Get the data for the train, validation and test sets.

        Returns
        -------
        Tuple containing the train, validation and test sets.
        """
        if self.test is None:
            raise ValueError('Test data not set up. Please call setup_dataset() first.')

        return self.test


def generate_month_path(path: str, date: date_type) -> str:
    """
    Generate the path for the month. The path is assumed to be in the format 'YYYY-MM'.

    Parameters
    ----------
    path : Path to the dataset directory.
    date : Date to generate the path for.

    Returns
    -------
    Path to the month.
    """
    year, month = date
    return os.path.join(path, f'{year}-{month}')

def load_data(month: str) -> DatasetTuple:
    """
    Load the data for a specific month. It is assumed that the folder contains X, y and z files, which can all be
    loaded into torch tensors as .pt files.  The X file is assumed to be the input variable, y is the output variable and z is the
    additional time-independent features.

    Parameters
    ----------
    month : Month for which to load the data. The month is assumed to be in the format 'YYYY-MM'.

    Returns
    -------
    Tuple of torch tensors as (X, y, z).
    """
    # Path to each file
    path_X = os.path.join(month, 'X.pt')
    path_y = os.path.join(month, 'y.pt')
    path_z = os.path.join(month, 'z.pt')

    X = torch.load(path_X, weights_only=False)
    y = torch.load(path_y, weights_only=False)
    z = torch.load(path_z, weights_only=False)

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

def combine_dataset(*datasets) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine the two datasets. It is assumed that the datasets are in format: X, y, z. The datasets are concatenated
    along the first dimension.


    Parameters
    ----------
    datasets : List of datasets to combine. Each dataset is a tuple of the form (X, y, z).

    Returns
    -------
    Combined dataset as a tensor.
    """
    first_dataset = datasets[0]
    if len(datasets) == 1:
        return first_dataset

    # Initialize the variables
    X, y, z = first_dataset
    for i in range(1, len(datasets)):
        X_i, y_i, z_i = datasets[i]

        # Concatenate the datasets
        X = torch.cat([X, X_i], dim=0)
        y = torch.cat([y, y_i], dim=0)
        z = torch.cat([z, z_i], dim=0)

    return X, y, z

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


def increment_month(year: int, month: int, increment_value: int) -> tuple[int, int]:
    """
    Increment the month by a given value. The month is incremented by the value and if it exceeds 12, the year is
    incremented by 1 and the month is set to 12 - increment_value.

    Parameters
    ----------
    year : Year to increment.
    month : Month to increment.
    increment_value : Value to increment the month by.

    Returns
    -------
    Year and month after incrementing.
    """
    month += increment_value
    if month > 12:
        year += 1
        month -= 12

    return year, month
import torch
import os
from typing import Tuple
import gc
import multiprocessing

date_type = list[int]

DatasetTuple = Tuple[torch.Tensor, torch.Tensor]

class BaseDatasetManager:
    def __init__(self, dataset_path: str, num_months: int, sequence_size: int=-1):
        """
        Initialize the dataset manager.

        Parameters
        ----------
        dataset_path : Path to the dataset directory. Assumed to contain subfolders which represent the data for each month
        num_months : Number of months to load
        sequence_size : Size of the sequence to keep. If -1, the entire sequence is kept.
        """
        self.dataset_path = dataset_path
        self.num_months = num_months
        self.sequence_size = sequence_size

        self.start = None
        self.end = None

        self.dataset: tuple[torch.Tensor, torch.Tensor] = torch.empty(0), torch.empty(0)

    def setup_dataset(self, start_month: str):
        """
        Set up the dataset by loading data from the specified directory.

        Parameters
        ----------
        start_month : Starting month for the dataset
        """
        self.start = split_month(start_month)
        year_start, month_start = self.start

        # Subtracted by 1 since first month is included in the dataset
        year_end, month_end = increment_month(year_start, month_start, self.num_months - 1)
        self.end = [year_end, month_end]

        # Generate all the months to loop over
        months = generate_dates(self.start, self.end)
        month_paths = [generate_month_path(self.dataset_path, month) for month in months]

        # Load all files
        results = []
        for month_path in month_paths:
            results.extend(load_data(month_path))
        self.dataset = combine_dataset(results)

    def increment_dataset(self):
        """
        Increment the dataset by adding one month.
        """

        # Completely clear the memory
        self.empty_dataset()

        # Recompute the start and end dates
        start_year, start_month = self.start
        incremented_year, incremented_month = increment_month(start_year, start_month, 1)

        self.setup_dataset(f'{incremented_year}-{incremented_month}')

    def empty_dataset(self):
        """
        Empty the dataset. This is used to clear the memory after the dataset has been used.
        """
        del self.dataset
        self.dataset = None
        torch.cuda.empty_cache()
        gc.collect()

    def get_dataset(self, normalize: bool=True, binary_classification: bool=True) -> DatasetTuple:
        """
        Get the dataset. The dataset is a tuple of the form (X, y). The X is the input variable and y is the output variable.

        Parameters
        ----------
        normalize : Boolean indicating whether to normalize the dataset. If True, the dataset is normalized.
        binary_classification : Boolean indicating whether the output variable is a binary classification problem.
        If True, the output variable is converted to a binary classification problem (0 or 1)

        Returns
        -------
        Tuple containing the input variable and the output variable.
        """
        if self.dataset is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        x, y = self.dataset

        # Normalizing the features
        if normalize:
            x, y = self.normalize()

        # Convert the output variable to a binary classification problem
        if binary_classification:
            y = convert_to_classification(y)

        # Ony keep sequence length
        if self.sequence_size > 0:
            x = x[:, -self.sequence_size:, :]

        return x, y

    def to_classification(self):
        """
        Convert the dataset to a binary classification problem. The output variable is converted to a binary classification
        problem (0 or 1). This is used for binary classification problems.
        """
        if self.dataset is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        # We do not want to override the original dataset
        y = self.dataset[1].clone()

        # Convert the output variable to a binary classification problem
        y = convert_to_classification(y)

        return y

    def normalize(self):
        """
        Normalize the dataset. The dataset is assumed to be a tuple of the form (X, y). The X is the input variable and y is the output variable.
        """
        raise NotImplementedError("Normalization not implemented in the base class. Please implement in the derived class.")

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

def load_data(month_path: str) -> list[DatasetTuple]:
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

        # Load the data
        X = torch.load(path_X, weights_only=False)
        y = torch.load(path_y, weights_only=False)

        # Make sure y is two dimensional
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        results.append((X, y))

    return results

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

def combine_dataset(datasets: list[DatasetTuple]) -> DatasetTuple:
    """
    Combine all the datasets for each month into a combined dataset. The datasets are assumed to be in the format

    Parameters
    ----------
    datasets : List of datasets to combine. Each dataset is a tuple of the form (X, y, z).

    Returns
    -------
    Combined dataset as a list of tuples.
    """
    Xs, ys = zip(*datasets)

    # Concatenate all the data
    X = torch.concatenate(Xs, dim=0)
    y = torch.concatenate(ys)

    return X, y

def convert_to_classification(y) -> torch.Tensor:
    """
    Convert the output of the dataset to a binary classification problem. The output is 1 if the value is greater than 0
    and 0 otherwise. This is used for binary classification problems.
    Parameters
    ----------
    y : Output variable to convert. The output is assumed to be a torch tensor.

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

def generate_dates(start_date: date_type, end_date: date_type) -> list[date_type]:
    """
    Generate a list of dates between the start and end date. The dates are given as [year, month].

    Parameters
    ----------
    start_date : Start date to process the trades from. Given as [year, month]
    end_date : End date to process the trades to. Given as [year, month]

    Returns
    -------
    List of dates between the start and end date
    """
    start_year, start_month = start_date
    end_year, end_month = end_date

    dates = []
    year, month = start_year, start_month

    # Checks if we are before the end year, or before the end month in the same year
    while (year < end_year) or (year == end_year and month <= end_month):
        dates.append([year, month])
        month += 1
        if month > 12:
            month = 1
            year += 1

    return dates
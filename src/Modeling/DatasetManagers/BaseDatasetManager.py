import torch
import os
from typing import Tuple
import gc

date_type = list[int]

DatasetTuple = Tuple[torch.Tensor, torch.Tensor]

class BaseDatasetManager:
    def __init__(self, dataset_path: str, num_months: int):
        """
        Initialize the dataset manager.

        Parameters
        ----------
        dataset_path : Path to the dataset directory. Assumed to contain subfolders which represent the data for each month
        num_months : Number of months to load
        """
        self.dataset_path = dataset_path
        self.num_months = num_months

        self.start = None
        self.end = None

        self.dataset: list[list[DatasetTuple]] = []

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
        self.dataset = [load_data(month_path) for month_path in month_paths]

    def increment_dataset(self):
        """
        Increment the dataset by adding one month.
        """

        # Remove the first month from the dataset
        del self.dataset[0]

        # Completely clear the memory
        clear_memory()

        # Recompute the start date
        start_year, start_month = self.start
        incremented_year, incremented_month = increment_month(start_year, start_month, 1)

        # Recompute the end date
        end_year, end_month = self.end
        incremented_end_year, incremented_end_month = increment_month(end_year, end_month, 1)

        # Update the start and end dates
        self.start = [incremented_year, incremented_month]
        self.end = [incremented_end_year, incremented_end_month]

        # Load data of the last month
        month_path = generate_month_path(self.dataset_path, self.end)
        self.dataset.append(load_data(month_path))

    def get_dataset(self) -> list[DatasetTuple]:
        """r
        Returns all the datasets concatenated to one list. Each element represents one day worth of data.

        Returns
        -------
        List of daily datasets. Each dataset is a tuple of the form (X, y).
        """
        if self.dataset is None:
            raise ValueError('Dataset not set up. Please call setup_dataset() first.')

        return combine_dataset(self.dataset)

    def clear_memory(self):
        """
        Empty the dataset. This is used to clear the memory after the dataset has been used.
        """
        torch.cuda.empty_cache()
        gc.collect()

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

def combine_dataset(datasets: list[list[DatasetTuple]]) -> list[DatasetTuple]:
    """
    Concatenate all the days of the month into a single dataset. The initial dataset is assumed to be of the form
    list[list[DatasetTuple]], indicating that each month contains a list of days, and each day contains a tuple of the
    form (X, y). After concatenation, the dataset is of the form list[DatasetTuple], indicating that each day is a
    tuple of the form (X, y).

    Parameters
    ----------
    datasets : List of datasets to combine. Each dataset is a tuple of the form (X, y, z).

    Returns
    -------
    Combined dataset as a list of tuples.
    """
    results = []
    for dataset in datasets:
        results.extend(dataset)

    return results

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
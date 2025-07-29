import torch
import os
from typing import Tuple

# Represents a tuple containing the data for a single dataset sample in the format (X, y, z_1, z_2, ...).
BaseSampleTuple = Tuple[torch.Tensor, torch.Tensor, ...]

class BaseDatasetManager:
    """
    Base class for dataset managers. This class is responsible for loading and managing datasets for each type
    of dataset (e.g., intraday, closing, etc.). It provides methods to set up the dataset, load data for specific
    days, and retrieve the training, validation, and test datasets.
    """
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset manager.

        Parameters
        ----------
        dataset_path : Path to the dataset directory. Assumed to contain subfolders which represent the data for each
        day.
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f'Dataset path {dataset_path} does not exist.')

        self.dataset_path = dataset_path
        self.dataset: list[list[BaseSampleTuple]] = [[], [], []]  # Assuming train, validation and test datasets

    def setup_dataset(self, train_size: int, validation_size: int, test_size: int, start_index: int = 0):
        """
        Set up the dataset by loading data from the specified directory.

        Parameters
        ----------
        train_size: Number of training examples to load
        validation_size: Number of validation examples to load
        test_size: Number of test examples to load. If not specified, the rest of the data will be used as the test set.
        start_index: Index to start loading data from. This is useful for resuming training from a specific point.
        """

        # Get all the dates in the dataset directory
        dates = sorted(os.listdir(self.dataset_path))[start_index:]
        if len(dates) == 0:
            raise ValueError(f'No data found in the dataset path {self.dataset_path}.')


        # Split the data into train, validation and test sets
        if train_size + validation_size + test_size > len(dates):
            raise ValueError(f'Sizes for train and validation sets exceed the number of available dates: {len(dates)}.')

        # Load the data for all dates
        dates_to_load = dates[:train_size + validation_size + test_size]
        data = self.load_dataset(dates_to_load)

        # Test size needs to be set to the remaining data if not specified
        if test_size == 0:
            test_size = len(dates) - (train_size + validation_size)

        # Make indices for splitting the data
        train_index = train_size
        validation_index = train_index + validation_size
        test_index = validation_index + test_size

        self.dataset[0] = data[:train_size]  # Training set
        self.dataset[1] = data[train_index:validation_index]  # Validation set
        self.dataset[2] = data[validation_index:test_index]  # Test set

    def load_dataset(self, dates: list[str]) -> list[BaseSampleTuple]:
        """
        Load the dataset from the specified directory. Almost exactly the same as the IntradayDatasetManager, but
        also loads in the exogenous features z, which are assumed to be time-independent features.

        Parameters
        ----------
        dates : List of dates to load. Each date is assumed to be in the format 'YYYY-MM'.

        Returns
        -------
        List of dataset samples. Each sample is a tuple of the form (X, y, z_1, z_2, ...).
        """

        # Used to store all the data loaded from the dataset
        data = []

        # Loading per day
        for date in dates:
            date_path = os.path.join(self.dataset_path, date)
            sample = self.load_single_day(date_path)
            data.append(sample)
        return data

    def load_single_day(self, date_path: str) -> BaseSampleTuple:
        """
        Load the data for a specific day. This is a base method that should be implemented in subclasses.

        Parameters
        ----------
        date_path : Path to the directory containing the data for a specific day.
        The day is assumed to be in the format 'YYYY-MM-DD'.

        Returns
        -------
        Variable number of torch tensors as (X, y, z_1, z_2, ...).
        """
        raise NotImplementedError('The load_single_day() method must be implemented in a subclass.')

    def get_train_dataset(self) -> list[BaseSampleTuple]:
        """
        Returns the training dataset.

        Returns
        -------
        List of training samples. Each dataset is a tuple of the form (X, y).
        """
        train_dataset = self.dataset[0]

        return train_dataset

    def get_validation_dataset(self) -> list[BaseSampleTuple]:
        """
        Returns the validation dataset.

        Returns
        -------
        List of validation samples. Each dataset is a tuple of the form (X, y).
        """
        validation_dataset = self.dataset[1]

        return validation_dataset

    def get_test_dataset(self) -> list[BaseSampleTuple]:
        """
        Returns the test dataset.

        Returns
        -------
        List of test samples. Each dataset is a tuple of the form (X, y).
        """
        test_dataset = self.dataset[2]

        return test_dataset
import torch
import os
import pandas as pd

class Dataset:
    def __init__(self, train_dir: str, validation_dir: str, test_dir: str):
        """
        Initialize the dataset object.

        Parameters
        ----------
        train_dir : Directory containing the training dataset.
        validation_dir : Directory containing the validation dataset.
        test_dir : Directory containing the test dataset.
        """
        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train = None
        self.y_test = None
        self.y_val = None

        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.test_dir = test_dir

    def get_train_data(self, horizon: int=0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the training data.

        Parameters
        ----------
        horizon : Number of time steps to predict.

        Returns
        -------
        Tuple containing the input and target variables.
        """
        if self.X_train is None:
            self.X_train, self.y_train = load_set(self.train_dir)

        return select_horizon(self.X_train, self.y_train, horizon)

    def get_validation_data(self, horizon: int=0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the validation data.

        Parameters
        ----------
        horizon : Number of time steps to predict.

        Returns
        -------
        Tuple containing the input and target variables.
        """
        if self.X_val is None:
            self.X_val, self.y_val = load_set(self.validation_dir)

        return select_horizon(self.X_val, self.y_val, horizon)

    def get_test_data(self, horizon: int=0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the test data.

        Parameters
        ----------
        horizon : Number of time steps to predict.

        Returns
        -------
        Tuple containing the input and target variables.
        """
        if self.X_test is None:
            self.X_test, self.y_test = load_set(self.test_dir)

        return select_horizon(self.X_test, self.y_test, horizon)

    def clear_memory(self):
        """
        Clear the memory of all loaded datasets.
        """
        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train = None
        self.y_test = None
        self.y_val = None

        torch.cuda.empty_cache()

def load_set(dataset_dir) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a dataset from a directory. The directory should contain the following parquet files: close.parquet,
    closing_returns.parquet, high.parquet, low.parquet, open.parquet, volume.parquet.
    Parameters
    ----------
    dataset_dir : Directory containing the dataset files.

    Returns
    -------
    Tuple containing the input and target variables.
    """

    X_names = ['close', 'high', 'low', 'open', 'volume']
    y_names = ['closing_returns']

    df_path = os.path.join(dataset_dir, X_names[0] + '.parquet')
    df = pd.read_parquet(df_path)
    # Check if there are any missing values in the dataset
    if df.isnull().values.any():
        raise ValueError('Dataset contains missing values')

    X = torch.tensor(df.values, dtype=torch.float32)
    for name in X_names[1:]:
        df_path = os.path.join(dataset_dir, name + '.parquet')
        df = pd.read_parquet(df_path)
        if df.isnull().values.any():
            raise ValueError('Dataset contains missing values')
        X = torch.dstack((X, torch.tensor(df.values, dtype=torch.float32)))

    df_path = os.path.join(dataset_dir, y_names[0] + '.parquet')
    df = pd.read_parquet(df_path)
    if df.isnull().values.any():
        raise ValueError('Dataset contains missing values')

    y = torch.tensor(df.values, dtype=torch.float32)

    return X, y

def select_horizon(X, y, horizon: int=0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select a horizon from the dataset.
    Parameters
    ----------
    X : Input variables.
    y : Target variables.
    horizon : Number of time steps to select.

    Returns
    -------
    Tuple containing the selected horizon from the input and target variables.
    """
    if horizon == 0:
        return X, y[:, -1].unsqueeze(1)
    else:
        return X[:, :-horizon, :], y[:, -1-horizon].unsqueeze(1)






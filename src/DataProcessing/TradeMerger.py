import pandas as pd
import multiprocessing
import os
import numpy as np
from numpy import ndarray
from pandas import Timestamp


class TradeMerger:
    """
    This class merges all the aggregrated trades into a single dataset, and then splits the dataset into training,
    validation and test sets.
    """
    def __init__(self, sequence_size: int, normalization_scheme: str, trades_dir: str, save_dir: str, dates: str, train_split: float,
                 val_split: float):
        """
        Initialize the TradeMerger class

        Parameters
        ----------
        sequence_size : Number of minutes in the sequence to consider
        normalization_scheme : Scheme to use for normalization
        trades_dir : Directory where all the trades are stored
        save_dir : Directory where all the results will be saved
        dates : File with all the trade dates that are present within the trades_dir
        train_split : Percentage of the dates to use for training
        val_split : Percentage of the dates to use for validation
        """
        self.sequence_size = sequence_size
        self.normalization_scheme = normalization_scheme
        self.trades_dir = trades_dir
        self.save_dir = save_dir

        dates = pd.read_csv(dates)
        self.dates = dates.values.flatten()
        n_dates = len(self.dates)

        train_index = int(n_dates * train_split)
        train_date = self.dates[train_index]
        self.train_date = pd.to_datetime(train_date)

        val_index = int(n_dates * (train_split + val_split))
        val_date = self.dates[val_index]
        self.val_date = pd.to_datetime(val_date)

        self.results = None
        self.datasets = {
            'train': {},
            'val': {},
            'test': {}
        }

    def add_stocks(self) -> None:
        """
        Add all the stocks in the trades_dir to the dataset

        Returns
        -------
        None
        """
        files = os.listdir(self.trades_dir)
        files.sort()
        items = [(os.path.join(self.trades_dir, file), self.sequence_size, self.normalization_scheme) for file in files]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(process_stock, items)

        self.results = combine_results(results)

    def split_data(self) -> None:
        """
        Split the data into training, validation and test sets

        Returns
        -------
        None
        """
        key_getter = lambda key: get_key(key, self.train_date, self.val_date)

        for date, data in self.results:
            key = key_getter(date)
            for column, values in data.items():
                if column not in self.datasets[key]:
                    self.datasets[key][column] = []

                self.datasets[key][column].append(values)

    def save_data(self) -> None:
        """
        Save the data to the save_dir

        Returns
        -------
        None
        """
        for key, data in self.datasets.items():
            save_path = os.path.join(self.save_dir, key)
            os.makedirs(save_path, exist_ok=True)

            for column, values in data.items():
                df = pd.DataFrame(values)
                df.to_parquet(os.path.join(save_path, column + '.parquet'), index=False)


def get_key(date: pd.Timestamp, train_date: pd.Timestamp, val_date: pd.Timestamp) -> str:
    """
    Get the key for the dataset based on the date

    Parameters
    ----------
    date : Date string in the format 'YYYY-MM-DD'
    train_date : Number of months in the training set
    val_date : Number of months in the validation set

    Returns
    -------
    Key for the dataset
    """
    if date < train_date:
        return 'train'
    elif date < val_date:
        return 'val'
    else:
        return 'test'

def z_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-normalization to the vwap column of a given series.

    Parameters
    ----------
    df : Series to normalize

    Returns
    -------
    Dataframe with vwap columns normalized
    """
    # These compute a rolling mean and standard deviation for the series based on all previous values
    mean = df['vwap'].expanding().mean()
    std = df['vwap'].expanding().std()
    std = std.fillna(1)

    df['vwap'] = (df['vwap'] - mean) / std

    return df

def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization to the vwap column of a given series.

    Parameters
    ----------
    df : Series to normalize

    Returns
    -------
    Dataframe with vwap columns normalized
    """
    # These compute the minimum and maximum values for the series
    min_value = df['vwap'].expanding().min()
    max_value = df['vwap'].expanding().max()

    df['vwap'] = (df['vwap'] - min_value) / (max_value - min_value)

    return df

def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log returns to the vwap column of a given series.

    Parameters
    ----------
    df : Series to apply log returns

    Returns
    -------
    Dataframe with vwap columns with log returns
    """
    # These compute the log returns for the series
    df['vwap'] = np.log(df['vwap'] / df['vwap'].shift(1))

    # Return at opening is just zero
    df = df.fillna(0)

    return df

def simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply simple returns to the vwap column of a given series.

    Parameters
    ----------
    df : Series to apply simple returns

    Returns
    -------
    Dataframe with vwap columns with simple returns
    """
    # These compute the simple returns for the series
    df['vwap'] = df['vwap'] / df['vwap'].shift(1) - 1

    # Return at opening is just zero
    df = df.fillna(0)

    return df

def relative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    This applies a normalization scheme to the volume column of a given series. It computes the relative volume
    which is measured relative to the current total volume of the day.

    Parameters
    ----------
    df : Series to apply relative volume

    Returns
    -------
    Dataframe with volume columns with relative volume
    """

    cumulative_volume = df['volume'].cumsum()
    df['volume'] = df['volume'] / cumulative_volume

    return df

def closing_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the log return of the closing auction relative to all other prices in the sequence.

    Parameters
    ----------
    df : DataFrame to compute the closing returns for

    Returns
    -------
    Log returns of the closing auction
    """

    vwap_prices = df['vwap']
    closing_price = df['vwap'].iloc[-1]
    returns = np.log(closing_price / vwap_prices)

    return returns

def normalize_vwap_and_volume(df: pd.DataFrame, normalization_scheme: str) -> pd.DataFrame:
    """
    Normalize the vwap and volume columns of a given series.

    Parameters
    ----------
    df : Series to normalize
    normalization_scheme : Scheme to use for normalization of vwap

    Returns
    -------
    Dataframe with vwap and volume columns normalized
    """

    # Normalizing the vwap column
    if normalization_scheme == 'z':
        df = z_normalize(df)
    elif normalization_scheme == 'min_max':
        df = min_max_normalize(df)
    elif normalization_scheme == 'log_returns':
        df = log_returns(df)
    elif normalization_scheme == 'simple_returns':
        df = simple_returns(df)
    else:
        raise ValueError('Invalid normalization scheme')

    # Normalizing the volume column
    df = relative_volume(df)

    return df

def process_file(df: pd.DataFrame, sequence_length: int, normalization_scheme: str) -> dict[str, np.ndarray]:
    """
    Process a file to get the data in the correct format.

    Parameters
    ----------
    df : DataFrame to process
    sequence_length : Number of minutes in the sequence
    normalization_scheme : Scheme to use for normalization

    Returns
    -------
    Dictionary with the processed data for each column
    """

    returns = closing_returns(df).values
    vwap_and_volume = normalize_vwap_and_volume(df, normalization_scheme)
    vwap = vwap_and_volume['vwap'].values
    volume = vwap_and_volume['volume'].values

    # For both we only need to consider data from the last sequence_length minutes
    returns = returns[-sequence_length - 1:-1]
    vwap = vwap[-sequence_length - 1:-1]
    volume = volume[-sequence_length - 1:-1]

    result = {'closing_returns': returns,
              'vwap': vwap,
              'volume': volume}

    return result

def process_stock(stock_path: str, sequence_length: int,
                  normalization_scheme: str) -> list[list[Timestamp | dict[str, ndarray]]]:
    """
    Process all trades of a given stock

    Parameters
    ----------
    stock_path : Path to the stock trades
    sequence_length : Number of minutes in the sequence
    normalization_scheme : Scheme to use for normalization

    Returns
    -------
    List with the processed data for each day
    """
    files = os.listdir(stock_path)
    if len(files) == 0:
        return []

    # Listdir does not return a sorted list
    files.sort()

    dataset = []
    for file in files:
        file_path = os.path.join(stock_path, file)
        date = file.split('.')[0]
        date = pd.to_datetime(date)

        df = pd.read_parquet(file_path)
        result = process_file(df, sequence_length, normalization_scheme)

        dataset.append([date, result])

    return dataset

def combine_results(results: list) -> list:
    """
    Combine multiple lists into a single list

    Parameters
    ----------
    results : List of lists to combine

    Returns
    -------
    List with all the elements from the input lists
    """
    combined = []
    for result in results:
        if result is not None:
            combined.extend(result)

    return combined
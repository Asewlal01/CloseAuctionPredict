from typing import Any

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
    def __init__(self, sequence_size: int, trades_dir: str, save_dir: str, dates: str, train_split: float,
                 val_split: float):
        """
        Initialize the TradeMerger class

        Parameters
        ----------
        sequence_size : Number of minutes in the sequence to consider
        trades_dir : Directory where all the trades are stored
        save_dir : Directory where all the results will be saved
        dates : File with all the trade dates that are present within the trades_dir
        train_split : Percentage of the dates to use for training
        val_split : Percentage of the dates to use for validation
        """
        self.sequence_size = sequence_size
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

    def add_socks(self) -> None:
        """
        Add all the stocks in the trades_dir to the dataset

        Returns
        -------
        None
        """
        files = os.listdir(self.trades_dir)
        files.sort()
        items = [(os.path.join(self.trades_dir, file), self.sequence_size) for file in files]
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
        for date, data in self.results:
            key = get_key(date, self.train_date, self.val_date)
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

def statistics(df: pd.DataFrame) -> dict[str, list[ndarray]]:
    """
    Get statistics for a DataFrame

    Parameters
    ----------
    df : DataFrame to get statistics for

    Returns
    -------
    Dictionary with statistics
    """
    stats = {}
    for column in df.columns:
        mean = df[column].expanding().mean()
        std = df[column].expanding().std()
        std = std.fillna(1)

        stats[column] = [mean.values, std.values]

    return stats

def process_file(df: pd.DataFrame, statistics, sequence_length: int) -> dict[str, np.ndarray]:
    """
    Process a file to get the data in the correct format. The data is normalized based on the previous day's statistics.

    Parameters
    ----------
    df : DataFrame to process
    previous_day_stats : Statistics from the previous day
    sequence_length : Number of minutes in the sequence

    Returns
    -------
    Dictionary with the processed data for each column
    """
    # We only need consider samples from the last sequence_length rows and the last row
    df = df.iloc[-sequence_length - 1:]

    # The return of the closing auction is measured relative to the last closing price at some interval
    close = df['close'].values
    closing_return = (close[-1] / close[:-1] - 1).copy()

    # # Normalization is based on the previous day's statistics
    # df = (df - previous_day_stats[0]) / previous_day_stats[1]

    # Removing the closing auction
    df = df.iloc[:-1]

    result = {'closing_returns': closing_return}
    for column in df.columns:
        mean = statistics[column][0][-sequence_length - 1:-1]
        std = statistics[column][1][-sequence_length - 1:-1]
        result[column] = (df[column].values - mean) / std

    return result

def process_stock(stock_path: str, sequence_size: int) -> list[list[Timestamp | dict[str, np.ndarray]]]:
    """
    Process all trades of a given stock

    Parameters
    ----------
    stock_path : Path to the stock trades
    sequence_size : Number of minutes in the sequence

    Returns
    -------
    List with the processed data for each day
    """
    files = os.listdir(stock_path)
    if len(files) == 0:
        return []

    # Files are not sorted
    files.sort()

    dataset = []
    max_returns = np.zeros(sequence_size)
    for file in files:
        file_path = os.path.join(stock_path, file)
        date = file.split('.')[0]
        date = pd.to_datetime(date)

        df = pd.read_parquet(file_path)
        current_stats = statistics(df.iloc[1:-1])
        result = process_file(df, current_stats, sequence_size)

        dataset.append([date, result])

        previous_day_stats = current_stats

    # Normalization of the closing returns
    # dataset = normalize_returns(dataset, max_returns, train_date)

    return dataset

def combine_results(results: list[Any]) -> list[Any]:
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
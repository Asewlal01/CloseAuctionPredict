import pandas as pd
import multiprocessing
import os
import numpy as np

class TradeMerger:
    def __init__(self, sequence_size: int, trades_dir, save_dir: str, train_months: int, val_months: int):
        self.sequence_size = sequence_size
        self.trades_dir = trades_dir
        self.save_dir = save_dir

        self.train_months = train_months
        self.val_months = val_months

        self.datasets = {
            'train': {},
            'val': {},
            'test': {}
        }


    def add_stock(self, stock_path: str):
        files = os.listdir(stock_path)
        previous_day = pd.read_parquet(os.path.join(stock_path, files[0]))
        for file in files[1:]:
            file_path = os.path.join(stock_path, file)
            date = file.split('.')[0]
            key = get_key(date, self.train_months, self.val_months)

            df = pd.read_parquet(file_path)
            result = process_file(df.copy(), previous_day, self.sequence_size)

            for column, values in result.items():
                if column not in self.datasets[key]:
                    self.datasets[key][column] = []

                self.datasets[key][column].append(values)

            previous_day = df

    def save_data(self):
        for key, data in self.datasets.items():
            save_path = os.path.join(self.save_dir, key)
            os.makedirs(save_path, exist_ok=True)

            for column, values in data.items():
                df = pd.DataFrame(values)
                df.to_parquet(os.path.join(save_path, column + '.parquet'), index=False)

def get_month(date: str) -> int:
    """
    Get the month from a date string

    Parameters
    ----------
    date : Date string in the format 'YYYY-MM-DD'

    Returns
    -------
    Month as an integer
    """
    return int(date.split('-')[1])

def get_key(date: str, train_months: int, val_months: int) -> str:
    """
    Get the key for the dataset based on the date

    Parameters
    ----------
    date : Date string in the format 'YYYY-MM-DD'
    train_months : Number of months in the training set
    val_months : Number of months in the validation set

    Returns
    -------
    Key for the dataset
    """
    month = get_month(date)
    if month < train_months:
        return 'train'
    elif month < train_months + val_months:
        return 'val'
    else:
        return 'test'

def statistics(df: pd.DataFrame) -> dict:
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
        mean = df[column].mean()
        std = df[column].std()
        stats[column] = [mean, std]

    return stats


def process_file(df: pd.DataFrame, previous_day: pd.DataFrame, sequence_length: int) -> dict[str, np.ndarray]:
    # The return of the closing auction is measured relative to the last closing price at some interval
    closing_price = df.iloc[-1, 0]
    closing_returns = closing_price / df['close'].iloc[-sequence_length - 1:-1] - 1

    # Normalization is based on the previous day's statistics
    previous_day_stats = statistics(previous_day.iloc[1:-1])
    for column in df.columns:
        mean, std = previous_day_stats[column]
        df[column] = (df[column] - mean) / std

    # Only leaving the last sequence_length rows without last row
    df = df.iloc[-sequence_length - 1:-1]

    result = {'closing_returns': closing_returns}
    for column in df.columns:
        result[column] = df[column].values

    return result













import os
from typing import Any

from Modeling.DatasetManagers.IntradayDatasetManager import generate_dates, date_type
import pandas as pd
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm


class IntradayFeatureEngineer:
    """
    This class handles the assembly of the intraday features from the limit order book data. It also groups the data
    for all stocks in the given month per day and saves the processed data in a specified directory.
    """

    def __init__(self, lob_path: str, save_path: str, sequence_size: int = 360, horizon: int = 5,
                 samples_to_keep: int = 1) -> None:
        """
        Initialize the IntradayDatasetAssembler with the path to the limit order book data.

        Parameters
        ----------
        lob_path : Path to the limit order book data directory.
        save_path : Path to the save directory.
        sequence_size : Size of the input sequence.
        horizon : Size of future sequence used to determine the target.
        samples_to_keep : Number of samples to keep for each date.
        """

        self.lob_path = lob_path
        self.stocks = os.listdir(lob_path)
        self.save_path = save_path
        self.sequence_size = sequence_size
        self.horizon = horizon
        self.samples_to_keep = samples_to_keep

    def assemble(self, start_date: date_type, end_date: date_type) -> None:
        """
        Assemble the intraday dataset.

        Parameters
        ----------
        start_date : Start date for the dataset as [year, month].
        end_date : End date for the dataset as [year, month].
        """

        # Generate all the months to loop over
        months = generate_dates(start_date, end_date)

        # Loop over all the months
        for month in tqdm(months):
            process_month(month, self.lob_path, self.save_path,
                          self.horizon, self.sequence_size, self.samples_to_keep)


def process_month(month: date_type, lob_path: str, save_path: str, horizon: int, sequence_size: int,
                  samples_to_keep: int) -> None:
    """
    Process the stocks in the given month.

    Parameters
    ----------
    month : Month to process as [year, month].
    lob_path : Path to the limit order book data directory.
    save_path : Path to save the processed data.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    samples_to_keep : Number of samples to keep
    """

    # Get all the files for the month
    files_to_process = get_files_to_process(month, lob_path)
    if not files_to_process:
        print(f"No files to process for month {month}.")
        return

    # Go through each date
    month_save_path = os.path.join(save_path, f'{month[0]}-{month[1]}')
    for date, files in files_to_process.items():
        process_day(date, files, month_save_path, horizon, sequence_size, samples_to_keep)


def get_files_to_process(month: date_type, lob_path: str, stocks_to_consider: list=None) -> dict[str, list[str]]:
    """
    Get the files for each stock in the given month to process.

    Parameters
    ----------
    month : Month to process as [year, month].
    lob_path : Path to the limit order book data directory.
    stocks_to_consider : List of stocks to consider. If None, all stocks in the directory will be considered.

    Returns
    -------
    List of file paths for the given month.
    """

    files_to_process = {}
    year, month = month

    # Convert to a date
    month = f"{year}-{month:02d}"

    # Using all the stocks
    if stocks_to_consider is None:
        # Get all the stocks in the directory
        stocks_to_consider = os.listdir(lob_path)

    for stock in stocks_to_consider:
        # Convert stock to string
        stock = str(stock) if not isinstance(stock, str) else stock
        stock_path = os.path.join(lob_path, stock)
        # Check if exists
        if not os.path.exists(stock_path):
            continue

        for date in os.listdir(stock_path):
            # Not the correct month
            if month not in date:
                continue

            # Checks if date is already in the dictionary
            if date not in files_to_process:
                files_to_process[date] = []

            date_path = os.path.join(stock_path, date)
            files_to_process[date].append(date_path)

    return files_to_process


def process_day(date: str, files: list[str], save_path: str, horizon: int, sequence_size: int,
                samples_to_keep: int) -> None:
    """
    Process the files for the given day.

    Parameters
    ----------
    date : Date to process.
    files : List of files to process.
    save_path : Path to the directory to save the processed data.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    samples_to_keep : Number of samples to keep for each date.
    """

    # Create a pool of workers
    items = [(file_path, horizon, sequence_size, samples_to_keep) for file_path in files]

    # Parallel processing of files
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_file, items)

    # Filter out empty results
    results = [result for result in results if result[0].size > 0 and result[1].size > 0]
    if not results:
        print(f"No valid data for date {date}.")
        return

    # Save the results
    tensor_save(results, save_path, date)


def process_file(file_path: str,
                 horizon: int, sequence_size: int, samples_to_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process the given file. It is assumed that each file is a parquet file.

    Parameters
    ----------
    file_path : Path to the file to process.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    samples_to_keep : Number of samples to keep from each stock at a given date.
    """

    # Load the data
    df = pd.read_parquet(file_path)

    x = construct_features(df, horizon, sequence_size, samples_to_keep)
    y = construct_labels(df, horizon, sequence_size, samples_to_keep)

    # Return if empty
    if x.size == 0 or y.size == 0:
        return np.array([]), np.array([])

    # Checks if the two arrays are the same length
    if x.shape[0] != y.shape[0]:
        print(f"File {file_path} has different lengths for x and y: {x.shape[0]} vs {y.shape[0]}")
        return np.array([]), np.array([])

    return x, y

def construct_features(df: pd.DataFrame, horizon: int, sequence_size: int, samples_to_keep: int) -> np.ndarray:
    """
    Construct the features for the given stock.

    Parameters
    ----------
    df : DataFrame containing the data.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    samples_to_keep : Number of samples to keep for each date.

    Returns
    -------
    Features for the given stock.
    """
    arr = df.values

    # The last few rows cannot be processed since we do not have enough data outside the window for computing the
    # return
    arr = arr[:-horizon]

    # Cannot process if we do not have enough data
    if arr.shape[0] < sequence_size:
        # print(f"File {df} has less than {sequence_size} rows.")
        return np.array([])

    # Conversions of prices to returns and volumes to relative volumes
    arr, mid_price = convert_to_returns(arr)
    arr, total_volume = convert_to_relative_volumes(arr)

    # Adding mid-price changes and total-volume changes
    mid_price_change = compute_mid_price_change(mid_price)
    total_volume_change = compute_total_volume_change(total_volume)
    changes = np.hstack((mid_price_change, total_volume_change)) # Creates 2D array with two columns
    arr = np.hstack((arr, changes)) # Concatenates the columns to the original array

    # Sliding window view to create sequences
    window_shape = (sequence_size, arr.shape[1])
    x = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_shape)[:, 0, :, :]

    # Only keep the last `samples_to_keep` samples
    if x.shape[0] > samples_to_keep:
        x = x[-samples_to_keep:]
    x = x.astype(np.float32)

    # Do not return x if there are NaN values
    if np.isnan(x).any():
        print("NaN values found in x")
        return np.array([])

    return x

def convert_to_returns(arr: np.array) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert prices in the given array to returns relative to the mid-price.

    Parameters
    ----------
    arr : Array to process containing price data.

    Returns
    -------
    Array with price returns computed and the mid-price.
    """
    # Get the mid-price
    mid_price = compute_mid_price(arr)

    # Obtaining the columns that contain the limit order book prices
    lob_columns = np.arange(0, 20, 2)
    prices = arr[:, lob_columns]

    # Convert prices to returns relative to the mid-price
    returns = np.log(prices / mid_price)

    # Replace the original prices with returns in the array
    arr[:, lob_columns] = returns

    return arr, mid_price

def compute_mid_price(x) -> np.ndarray:
    """
    Compute the mid-price from the limit order book data.

    Parameters
    ----------
    x : Input array

    Returns
    -------
    Mid-price at each time step.
    """
    best_bid = x[:, 0]  # Best bid price
    best_ask = x[:, 10]  # Best ask price

    mid_price = (best_bid + best_ask) / 2.0

    # Convert to 2d
    mid_price = mid_price[:, np.newaxis]

    return mid_price

def convert_to_relative_volumes(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert volumes in the given array to relative volumes.

    Parameters
    ----------
    arr : Array to process containing volume data.

    Returns
    -------
    Array with relative volumes computed and the total volume.
    """
    # Get the volume columns
    volume_columns = np.arange(1, 20, 2)

    # Compute the total volume for each sample
    total_volume = compute_total_volume(arr)

    # Convert volumes to relative volumes
    arr[:, volume_columns] = arr[:, volume_columns] / total_volume

    return arr, total_volume

def compute_total_volume(arr: np.ndarray) -> np.ndarray:
    """
    Compute the total volume from the limit order book data.

    Parameters
    ----------
    arr : Input array

    Returns
    -------
    Total volume at each time step
    """
    # Get the volume columns
    volume_columns = np.arange(1, 20, 2)

    # Compute the total volume for each sample
    total_volume = np.sum(arr[:, volume_columns], axis=1, keepdims=True)

    return total_volume

def compute_mid_price_change(mid_price: np.ndarray) -> np.ndarray:
    """
    Compute the mid-price change from the limit order book data.

    Parameters
    ----------
    mid_price : Mid-price array

    Returns
    -------
    Mid-price change for each sample.
    """

    # Compute the change in mid-price
    mid_price_change = np.zeros_like(mid_price)
    mid_price_change[1:] = np.log(mid_price[1:] / mid_price[:-1])

    return mid_price_change

def compute_total_volume_change(total_volume) -> np.ndarray:
    """
    Compute the total volume change from the limit order book data.

    Parameters
    ----------
    total_volume: Total volume array

    Returns
    -------
    Total volume change for each sample.
    """

    # Compute the change in total volume
    total_volume_change = np.zeros_like(total_volume)
    total_volume_change[1:] = np.log(total_volume[1:] / total_volume[:-1])

    return total_volume_change

def construct_labels(df: pd.DataFrame, horizon: int, sequence_size: int, samples_to_keep: int) -> np.ndarray:
    """
    Construct the labels for the given stock.

    Parameters
    ----------
    df : DataFrame containing the data.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    samples_to_keep : Number of samples to keep for each date.

    Returns
    -------
    y : Labels for the given stock.
    """
    # Cannot be processed
    if df.empty or df.shape[0] < sequence_size:
        return np.array([])

    # The mid-price is the average of the bid and ask prices
    mid_price = (df['ba1'] + df['bb1']) / 2

    # Y is average mid point price change using horizon points in the past and horizon points in the future
    averages = mid_price.rolling(window=horizon).mean()
    average_future = averages.shift(-horizon)

    # Compute the change
    y = np.log(average_future / averages)

    # The first few rows cannot be used due to sequence size window
    y = y[sequence_size - 1:]

    # And last few rows are NaN because of future window being out of bounds
    y = y[:-horizon]

    if y.shape[0] > samples_to_keep:
        y = y[-samples_to_keep:]

    y = y.astype(np.float32)
    if np.isnan(y).any():
        print("NaN values found in y")
        return np.array([])

    return y

def tensor_save(results, save_path: str, date: str) -> None:
    """
    Save the results  as a tensor after processing them.

    Parameters
    ----------
    results : List of tuples containing the results.
    save_path : Path to the directory to save the processed data.
    date : Date to save the data for.
    """

    # Split into X and y
    xs, ys = zip(*results)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys)

    # To tensors
    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)

    # Remove extensions from the date
    date = date.split('.')[0]

    # Save the results
    date_path = os.path.join(save_path, date)
    os.makedirs(date_path, exist_ok=True)

    x_save_path = os.path.join(date_path, 'x.pt')
    y_save_path = os.path.join(date_path, 'y.pt')
    torch.save(xs, x_save_path)
    torch.save(ys, y_save_path)

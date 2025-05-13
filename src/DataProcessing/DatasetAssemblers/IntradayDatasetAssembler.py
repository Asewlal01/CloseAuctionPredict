import os
from Modeling.DatasetManager import generate_dates, date_type
import pandas as pd
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm

class IntradayDatasetAssembler:
    """
    Assembles intraday datasets for analysis.
    """

    def __init__(self, lob_path: str, save_path: str, sequence_size: int=360, horizon: int=5,
                 samples_to_keep: int=1) -> None:
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
            process_month(month, self.lob_path, self.save_path, self.horizon, self.sequence_size, self.samples_to_keep)

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

def get_files_to_process(month: date_type, lob_path: str) -> dict[str, list[str]]:
    """
    Get the files for each stock in the given month to process.

    Parameters
    ----------
    month : Month to process as [year, month].
    lob_path : Path to the limit order book data directory.

    Returns
    -------
    List of file paths for the given month.
    """
    files_to_process = {}
    year, month = month

    # Convert to a date
    month = f"{year}-{month:02d}"

    for stock in os.listdir(lob_path):
        stock_path = os.path.join(lob_path, stock)
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
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_file, items)

    # Filter out empty results
    results = [result for result in results if result[0].size > 0 and result[1].size > 0]
    if not results:
        print(f"No valid data for date {date}.")
        return

    # Save the results
    tensor_save(results, save_path, date)

def process_file(file_path: str, horizon: int, sequence_size: int, samples_to_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process the given file. It is assumed that each file is a parquet file.

    Parameters
    ----------
    file_path : Path to the file to process.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    """
    # Load the data
    df = pd.read_parquet(file_path)

    x = construct_features(df, horizon, sequence_size, samples_to_keep)
    y = construct_labels(df, horizon, sequence_size, samples_to_keep)

    # Checks if the two arrays are the same length
    if x.shape[0] != y.shape[0]:
        print(f"File {file_path} has different lengths for x and y: {x.shape[0]} vs {y.shape[0]}")
        return np.array([]), np.array([])

    return x, y

def construct_features(df: pd.DataFrame, horizon: int, sequence_size: int, samples_to_keep: int) -> np.ndarray:
    arr = df.values

    # The last few rows cannot be processed since we do not have enough data outside the window for computing the
    # return
    arr = arr[:-horizon]

    # Cannot process if we do not have enough data
    if arr.shape[0] < sequence_size:
        # print(f"File {df} has less than {sequence_size} rows.")
        return np.array([])

    x = np.lib.stride_tricks.sliding_window_view(arr, window_shape=(sequence_size, arr.shape[1]))[:, 0, :, :]

    if x.shape[0] > samples_to_keep:
        x = x[-samples_to_keep:]

    x = x.astype(np.float32)

    if np.isnan(x).any():
        print("NaN values found in x")
        return np.array([])

    return x

def construct_labels(df: pd.DataFrame, horizon: int, sequence_size: int, samples_to_keep: int) -> np.ndarray:
    """
    Construct the labels for the given stock.

    Parameters
    ----------
    df : DataFrame containing the data.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.

    Returns
    -------
    y : Labels for the given stock.
    """
    if df.empty or df.shape[0] < sequence_size:
        # print(f"File {df} is empty or has less than {sequence_size} rows.")
        return np.array([])

    # The mid price is the average of the bid and ask prices
    mid_price = (df['ba1'] + df['bb1']) / 2

    # Y is average mid point price change using horizon points in the past and horizon points in the future
    averages = mid_price.rolling(window=horizon).mean()
    average_future = averages.shift(-horizon)

    # Compute the change
    y = average_future / averages - 1

    # The first few rows cannot be used due to sequence size window
    # And last few rows are NaN because of future window being out of bounds
    y = y[sequence_size-1:-horizon]

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
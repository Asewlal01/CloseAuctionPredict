import os
from Modeling.DatasetManagers.IntradayDatasetManager import date_type, generate_dates
import pandas as pd
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm
from DataProcessing.DatasetAssemblers.IntradayDatasetAssembler import get_files_to_process
from DataProcessing.DatasetAssemblers.LobDatasetAssembler import (get_idx_and_day, get_auction_data)
from datetime import datetime

class ClosingDatasetAssembler:
    """
    Assembles Closing Auction datasets for analysis.
    """

    def __init__(self, lob_path: str, trade_path: str, auction_path: str, save_path: str, sequence_size: int=360,
                 horizon: int=5) -> None:
        """
        Initialize the ClosingDatasetAssembler.

        Parameters
        ----------
        lob_path : Path to the limit order book data directory.
        trade_path : Path to the trade data directory.
        auction_path : Path to the auction data directory.
        save_path : Path to save the processed data.
        sequence_size : Size of the input sequence.
        horizon : Size of future sequence used to determine the target.
        """
        self.lob_path = lob_path
        self.trade_path = trade_path
        self.auction_path = auction_path

        self.save_path = save_path
        self.sequence_size = sequence_size
        self.horizon = horizon

    def assemble_dataset(self, start_date: date_type, end_date: date_type) -> None:
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
            process_month(month,
                          self.lob_path, self.trade_path, self.auction_path, self.save_path,
                          self.horizon, self.sequence_size)

def process_month(month: date_type,
                  lob_path: str, trade_path: str, auction_path: str, save_path: str,
                  horizon: int, sequence_size: int) -> None:
    """
    Process all files for a given month.

    Parameters
    ----------
    month : Month to process as [year, month].
    lob_path : Path to the limit order book data directory.
    trade_path : Path to the trade data directory.
    auction_path : Path to the auction data directory.
    save_path : Path to save the processed data.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    """
    # Get all the files for the month
    files_to_process = get_files_to_process(month, lob_path)
    if not files_to_process:
        print(f"No files to process for month {month}.")
        return

    # Save path needs to exist to save the data
    month_save_path = os.path.join(save_path, f'{month[0]}-{month[1]}')
    os.makedirs(month_save_path, exist_ok=True)

    # Go through all the dates
    for date, files in files_to_process.items():
        process_day(date, files, auction_path, trade_path, month_save_path, horizon, sequence_size)

def process_day(date: str, files: list[str], auction_path: str, trade_path: str, save_path: str,
                horizon: int, sequence_size: int) -> None:
    """
    Process all files for a given day.

    Parameters
    ----------
    date : Date to process
    files : List of files to process
    auction_path : Path to auction data directory
    trade_path : Path to trade data directory
    save_path : Path to save the processed data
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    """
    items = [(file_path, auction_path, trade_path, horizon, sequence_size) for file_path in files]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_file, items)

    # Filtering
    filtered_results = []
    for result in results:
        x, y, z = result
        if x.size > 0 and y.size > 0 and z.size > 0:
            filtered_results.append((x, y, z))

    if not filtered_results:
        return

    # Saving
    tensor_save(filtered_results, save_path, date, horizon)

def process_file(file_path: str, auction_path: str, trade_path: str,
                 horizon: int, sequence_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the given file. It is assumed that each file is a parquet file. Empty arrays are returned if the data
    is empty or if there are not enough rows.

    Parameters
    ----------
    file_path : Path to the file to process.
    auction_path : Path to the auction data directory.
    trade_path : Path to the trade data directory.
    horizon : Number of points in the future to use for the target variable.
    sequence_size : Size of the input sequence.
    """
    # Load the data
    lob_df = pd.read_parquet(file_path)
    idx, day = get_idx_and_day(file_path)

    # Before any processing, we check if the data is empty or if there are not enough rows
    if lob_df.empty or lob_df.shape[0] < sequence_size:
        return np.array([]), np.ndarray([]), np.ndarray([])

    x = construct_features(lob_df, trade_path, day, idx, sequence_size)
    y = construct_labels(lob_df, auction_path, day, idx, horizon)
    z = construct_exogenous(day, idx, auction_path, file_path, horizon)

    return x, y, z

def construct_features(lob_df: pd.DataFrame, trade_path: str, day: str, idx: str, sequence_size: int) -> np.ndarray:
    """
    Construct the features for the model. Returns an empty array if the trade data is empty or if there are not
    enough rows.

    Parameters
    ----------
    lob_df : DataFrame containing the limit order book data.
    trade_path : Path to the trade data directory.
    day : Day to get the data for.
    idx : Stock index.
    sequence_size : Size of the input sequence.

    Returns
    -------
    Feature set for the model.
    """
    # Get the trade data
    trade_df = get_trade_data(idx, day, trade_path)

    # If the trade data is empty, we cannot create the features
    if trade_df.empty:
        return np.array([])

    # Concat the two dataframes
    df = pd.concat([lob_df, trade_df], axis=1)
    arr = df.values

    # Not enough data to create the features
    if arr.shape[0] < sequence_size:
        return np.array([])

    # Only keep the sequence_size last rows
    x = arr[-sequence_size:]
    x = x.astype(np.float32)

    # Checking for nans
    if np.isnan(x).any():
        return np.array([])

    return x

def get_trade_data(idx, day, trade_path):
    """
    Get the trade data for a given stock and day.

    Parameters
    ----------
    idx : Stock index.
    day : Day to get the data for.
    trade_path : Path to the trade data directory.
    """
    file_path = os.path.join(trade_path, idx, f'{day}.parquet')
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    return df

def construct_labels(lob_df: pd.DataFrame, auction_path: str, day: str, idx: str, horizon: int) -> np.ndarray:
    """
    Construct the labels for the model. Returns an empty array if the auction data is empty or if there are not
    enough rows.

    Parameters
    ----------
    lob_df : DataFrame containing the limit order book data.
    auction_path : Path to the auction data directory.
    day : Day to get the data for.
    idx : Stock index.
    horizon : Prediction horizon, or averaging window which is used to determine current price.

    Returns
    -------
    The label, which is the log return of the closing price over the average price.
    """
    # Get the auction data
    auction_data = get_auction_data(idx, day, auction_path)

    # Empty auction data
    if auction_data.empty:
        return np.array([])

    # Get the closing price
    closing_row = auction_data.loc['close']
    closing_price = closing_row['Price']

    # Determine the mid-price of the last horizon rows
    horizon_rows = lob_df.iloc[-horizon:]
    mid_price = (horizon_rows['ba1'] + horizon_rows['bb1']) / 2
    average_price = mid_price.mean()

    # Compute the change
    y = np.log(closing_price / average_price)

    return np.array([y])

def construct_exogenous(day: str, idx: str, auction_path: str, file_path: str, horizon: int) -> np.ndarray:
    """
    Construct the exogenous features for the model. This is based on the previous day data, and today's opening
    price. The exogenous features is yesterday's return and the overnight return (opening price relative to yesterday's
    closing price).

    Parameters
    ----------
    day : Day to get the data for.
    idx : Stock index.
    auction_path : Path to the auction data directory.
    file_path : Path to the file to process of today
    horizon : Number of points in the future to use for the target variable.

    Returns
    -------
    Exogenous features for the model as an array with 2 elements.
    """
    # We first need to know the previous day
    previous_day = get_previous_day(day)

    # Get the auction data for the previous day and return empty array if it is empty
    previous_auction_data = get_auction_data(idx, previous_day, auction_path)
    if previous_auction_data.empty:
        return np.array([])

    # Get the auction data for today
    today_auction_data = get_auction_data(idx, day, auction_path)
    if today_auction_data.empty:
        return np.array([])


    # Get the closing price for the previous day and today's opening price
    previous_closing_row = previous_auction_data.loc['close']
    previous_closing_price = previous_closing_row['Price']
    today_opening_row = today_auction_data.loc['open']
    today_opening_price = today_opening_row['Price']
    overnight_return = np.log(today_opening_price / previous_closing_price)

    # Previous day return requires the previous day data
    previous_day_path = previous_file_path(file_path, previous_day)
    # Need to check if the file exists
    if not os.path.exists(previous_day_path):
        return np.array([])
    previous_lob_df = pd.read_parquet(previous_day_path)

    # Cannot create the features if the data is empty or if there are not enough rows
    if previous_lob_df.empty or previous_lob_df.shape[0] < horizon:
        return np.array([])

    previous_day_return = construct_labels(previous_lob_df, auction_path, previous_day, idx, horizon)
    if previous_day_return.size == 0:
        return np.array([])

    exogenous_features = np.append(previous_day_return, overnight_return)
    return exogenous_features


def get_previous_day(day: str) -> str:
    """
    Get the previous day for a given day. It is assumed that we only use working days.

    Parameters
    ----------
    day : Day to get the previous day for.

    Returns
    -------
    Previous day.
    """
    date = datetime.strptime(day, '%Y-%m-%d').date()
    if date.weekday() == 0:
        # If it's Monday, go back to Friday
        date = date - pd.DateOffset(days=3)
    else:
        # Otherwise, go back one day
        date = date - pd.DateOffset(days=1)
    return date.strftime('%Y-%m-%d')

def previous_file_path(today_file_path: str, previous_day: str) -> str:
    """
    Get the previous file path for a given file path. It is assumed that the file path is in the format
    path/idx/day.parquet.

    Parameters
    ----------
    today_file_path : Path to the file for today.
    previous_day : Previous day to get the file path for.

    Returns
    -------
    File path of previous day.
    """
    lob_path = os.path.dirname(today_file_path)
    previous_day_path = os.path.join(lob_path, f'{previous_day}.parquet')

    return previous_day_path

def tensor_save(results: list[tuple[np.ndarray, np.ndarray, np.ndarray]], save_path: str, date: str, horizon) -> None:
    """
    Save the results to a tensor file.

    Parameters
    ----------
    results : List of tuples containing the features and labels.
    save_path : Path to save the processed data.
    date : Date to save the data for.
    horizon : Prediction horizon
    """
    xs, ys, zs = zip(*results)
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    z = np.stack(zs, axis=0)

    # Normalize
    x, z = normalize_features(x, z, horizon)

    # Filter out results
    x, y, z = filter_outliers(x, y, z)

    # Converting to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)

    # Remove extensions from the date
    date = date.split('.')[0]

    # Path to save the data
    date_path = os.path.join(save_path, date)
    os.makedirs(date_path, exist_ok=True)

    # Save path of each tensor
    x_save_path = os.path.join(date_path, 'x.pt')
    y_save_path = os.path.join(date_path, 'y.pt')
    z_save_path = os.path.join(date_path, 'z.pt')

    # Save the tensors
    torch.save(x, x_save_path)
    torch.save(y, y_save_path)
    torch.save(z, z_save_path)

def normalize_features(x: np.ndarray, z: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the features.

    Parameters
    ----------
    x : Features to normalize.
    z : Exogenous features to normalize.
    horizon : Number of points in the future to use for the target variable.

    Returns
    -------
    Normalized features.
    """
    # Normalize the prices
    x = normalize_prices(x, horizon)

    # Normalize the volumes
    x = normalize_volumes(x, horizon)

    # Normalize traded volume
    x = normalize_traded_volume(x, horizon)

    # Normalize exogenous features
    z = normalize_exogenous(z)

    return x, z

def normalize_prices(x: np.ndarray, horizon: int) -> np.ndarray:
    """
    Normalize the prices.

    Parameters
    ----------
    x : Prices to normalize.
    horizon : Number of points in the future to use for the target variable.

    Returns
    -------
    Normalized prices.
    """
    # Lob price columns are first even 20 columns
    lob_price_columns = np.arange(0, 20, 2)

    # The last 5 excluding the last one are trade prices (open, close, high, low, vwap)
    trade_price_columns = np.arange(20, 25)

    # Combining them
    price_columns = np.concat([lob_price_columns, trade_price_columns], axis=0)

    # The minimum price is based on lowest bid
    lowest_bid = x[:, :-horizon, lob_price_columns[4]]
    min_lowest_bid = lowest_bid.min(axis=1, keepdims=True)

    # The maximum price is based on highest ask
    highest_ask = x[:, :-horizon, lob_price_columns[9]]
    max_highest_ask = highest_ask.max(axis=1, keepdims=True)

    # Both arrays need to be 3D for proper broadcasting
    min_price = min_lowest_bid[:, np.newaxis, :]
    max_price = max_highest_ask[:, np.newaxis, :]

    # Normalization
    x[:, :, price_columns] = (x[:, :, price_columns] - min_price) / (max_price - min_price)

    return x

def normalize_volumes(x: np.ndarray, horizon: int) -> np.ndarray:
    """
    Normalize the volume features using min-max normalization. It uses all the volume features to normalize the

    Parameters
    ----------
    x: Volume features to normalize.
    horizon: Number of points in the future to use for the target variable.

    Returns
    -------

    """
    # Volume columns are uneven columns
    volume_columns = np.arange(1, 20, 2)

    # Min-Max is based on all the columns
    min_volume = x[:, :-horizon, volume_columns].min(axis=(1, 2), keepdims=True)
    max_volume = x[:, :-horizon, volume_columns].max(axis=(1, 2), keepdims=True)

    # Normalize
    x[:, :, volume_columns] = (x[:, :, volume_columns] - min_volume) / (max_volume - min_volume)

    return x

def normalize_traded_volume(x: np.ndarray, horizon: int) -> np.ndarray:
    """
    Normalize the traded volume.

    Parameters
    ----------
    x : Traded volume to normalize.
    horizon : Number of points in the future to use for the target variable.

    Returns
    -------
    Normalized traded volume.
    """
    # The traded volume is the last column
    traded_volume = x[:, :-horizon, -1]

    # The minimum and maximum traded volume
    min_traded_volume = traded_volume.min(axis=1, keepdims=True)
    max_traded_volume = traded_volume.max(axis=1, keepdims=True)

    # Normalization
    x[:, :, -1] = (x[:, :, -1] - min_traded_volume) / (max_traded_volume - min_traded_volume)

    return x

def normalize_exogenous(z: np.ndarray) -> np.ndarray:
    """
    Normalize the exogenous features.

    Parameters
    ----------
    z : Exogenous features to normalize.

    Returns
    -------
    Normalized exogenous features.
    """
    # Get min and max
    min_z = z.min(axis=0, keepdims=True)
    max_z = z.max(axis=0, keepdims=True)

    # Normalization
    z = (z - min_z) / (max_z - min_z)

    return z

def filter_outliers(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out outliers from the dataset. The outliers are samples which contains values smaller than 0 or greater than
    1 in the lower lob levels.

    Parameters
    ----------
    x : Features to filter.
    y : Labels
    z : Exogenous features

    Returns
    -------
    Filtered features, labels and exogenous features.
    """
    # Lob price columns are first even 20 columns
    lob_price_columns = np.arange(0, 20, 2)

    # Check if the values are between 0 and 1
    mask = np.all((x[:, :, lob_price_columns] >= 0) & (x[:, :, lob_price_columns] <= 1), axis=(1, 2))

    # Filter the data
    x = x[mask]
    y = y[mask]
    z = z[mask]

    return x, y, z

def get_idx_and_day(file_path: str) -> tuple[str, str]:
    """
    Get the index and day from the file path.
    Parameters
    ----------
    file_path

    Returns
    -------

    """
    # Get the index and day from the file path
    parts = file_path.split(os.sep)
    idx = parts[-2]
    day = parts[-1]

    # Remove extension
    day = day.split('.')[0]

    return idx, day

def get_auction_data(idx: str, day: str, auction_path: str) -> pd.DataFrame:
    """
    Get the auction data for the given index and day.
    Parameters
    ----------
    idx : Index to get the auction data for.
    day : Day to get the auction data for.
    auction_path : Path to the auction data directory.

    Returns
    -------
    DataFrame with the auction data.
    """
    # Get the path to the auction data
    path = os.path.join(auction_path, idx, f'{day}.parquet')

    # Check if the file exists
    if not os.path.exists(path):
        return pd.DataFrame()

    # Load the data
    df = pd.read_parquet(path)
    return df
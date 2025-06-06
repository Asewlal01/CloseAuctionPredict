from DataProcessing.Processors import BaseProcessor
from DataProcessing.Processors import LimitOrderBookProcessor
import pandas as pd
import numpy as np
import datetime
import os
import multiprocessing

class TradeProcessor(BaseProcessor):
    """
    This class processes trade data files. It inherits from the FileProcessor class and implements the process_file method.
    The trade data is assumed to be a gzip compressed CSV file.
    """
    def __init__(self, file_path: str, lob_data: str, max_price):
        """
        Initialize the TradeProcessor with the path to the file to be processed and the interval for aggregation.
        The file is loaded and the 't' column is converted to datetime format.

        Parameters
        ----------
        file_path : Path to the file to be processed.
        lob_data : Path to the processed lob data.
        """
        self.aggregated_data = None
        self.lob_data = lob_data
        self.max_price = max_price
        super().__init__(file_path)

    def process_file(self):
        """
        Process the trade data file. This method is called during initialization.
        """
        self.aggregated_data = process_all_days(self.df, self.lob_data, self.max_price)

    def save_results(self, save_path):
        # Check if the aggregated data is empty
        if self.aggregated_data is None:
            print("Aggregated data is empty. Cannot save results.")
            return

        os.makedirs(save_path, exist_ok=True)
        for date, df in self.aggregated_data.items():
            # Create the file name
            file_name = f"{date}.parquet"
            file_path = os.path.join(save_path, file_name)

            # Save the dataframe to a parquet file
            df.to_parquet(file_path)


def process_all_days(df: pd.DataFrame, lob_data: str, max_price: int) -> dict[datetime.date, pd.DataFrame]:
    """
    Process all the trades of a given dataframe. All trades within this day are aggregated to a given interval.
    The opening and closing auctions are also included in the aggregated data, with the first row being the opening auction
    and the last row being the closing auction.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    lob_data : Path to the processed lob data.

    Returns
    -------
    Dictionary with dates as keys and DataFrames as values. Each DataFrame contains the aggregated trade data for that date.
    """
    # Process the dataframe
    items = groups_to_process(df, lob_data, max_price)
    if not items:
        return {}
    with multiprocessing.Pool() as pool:
        processed_groups = pool.starmap(process_day, items)

    # Concatenate the results into a dictionary
    processed_groups = {date: data for date, data in processed_groups if not data.empty}

    return processed_groups

def groups_to_process(df: pd.DataFrame, lob_data: str, max_price) -> list[tuple[pd.DataFrame, datetime.date, pd.Index]]:
    """
    This function returns a list of tuples containing the dataframes to be processed and their corresponding dates.

    Parameters
    ----------
    df : DataFrame containing the trade data. It is assumed that the 't' column is in datetime format.
    lob_data : Path to the processed limit order book data. This is used to get the indices for the trades.

    Returns
    -------
    All the groups of trades that can be processed. Each group is a tuple containing the dataframe,
    the date and the indices of the limit order book data.
    """

    # Check if lob_data is a valid directory
    if not os.path.isdir(lob_data):
        return []

    groups = df.groupby(df['t'].dt.date)

    items = []
    for date, group in groups:
        str_date = str(date)

        lob_data_path = os.path.join(lob_data, str_date)
        lob_data_path = f'{lob_data_path}.parquet'
        if not os.path.isfile(lob_data_path):
            continue

        # Load the lob dataframe
        lob_df = pd.read_parquet(lob_data_path)

        # We only need the indices of the lob data
        lob_indices = lob_df.index

        items.append((group, date, lob_indices, max_price))

    return items

def process_day(df: pd.DataFrame, date: datetime.date, lob_indices: pd.Index, max_price: int) -> tuple[datetime.date, pd.DataFrame]:
    """
    Process all the trades of a given day. All trades within this day are aggregated to a given interval.
    The opening and closing auctions are also included in the aggregated data, with the first row being the opening auction
    and the last row being the closing auction.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    date : Date of the trades to be processed.
    lob_indices : Indices of the limit order book data.
    max_price : Maximum price for the trades. This is used to filter out trades that are not relevant.

    Returns
    -------
    DataFrame containing the aggregated trade data.
    """

    # We only need NORMAL trades for processing
    df = df[df['flag'] == 'NORMAL']
    if df.empty:
        return date, pd.DataFrame()

    # Check if there is a price greater than the max price. This is likely an error in the data.
    if (df['price'] > max_price).any():
        return date, pd.DataFrame()

    # We need to add the first time stamp to the indices if it is before the first lob index
    first_timestamp = df['t'].iloc[0]
    if first_timestamp < lob_indices[0]:
        first_timestamp -= pd.Timedelta(seconds=1) # To avoid the first timestamp being equal to the first lob index
        lob_indices = pd.Index([first_timestamp] + lob_indices.tolist())

    # Bin the dataframe to the lob indices
    binned = pd.cut(df['t'], bins=lob_indices)

    aggregated_trades = aggregate_trades(df, binned)
    if aggregated_trades.empty:
        return date, pd.DataFrame()

    # Fix missing values
    aggregated_trades = fix_missing(aggregated_trades)

    return date, aggregated_trades

def aggregate_trades(df: pd.DataFrame, binned: pd.DataFrame) -> pd.DataFrame:
    """
    This function aggregates all the trades of a given df using a resampling interval.

    Parameters
    ----------
    df : Dataframe to be aggregated. Assumed to have columns 'price' and 'quantity'.
    binned : Binned dataframe to be used for aggregation. Assumed to have a column 't' with the time of the trades.

    Returns
    -------
    Dataframe of resampled trade data
    """

    # Group the trades by the time bins
    groups = df.groupby(binned, observed=False)
    aggregated_trades = groups.apply(aggregate_interval)

    if aggregated_trades.empty:
        return pd.DataFrame()

    # We want a timestamp as the index instead of intervals
    aggregated_trades.index = [interval.right for interval in aggregated_trades.index]

    # Check if unique
    if not aggregated_trades.index.is_unique:
        raise ValueError("Timestamp indices have non-unique indices")

    return aggregated_trades

def aggregate_interval(df: pd.DataFrame) -> pd.Series:
    """
    This function aggregates the trades and quantities within a dataframe to a single row. This row contains the
    open, high, low, close and vwap prices and the standard deviation of the prices, referred to as 'vwps'. Additionally,
    the total quantity of trades is also included in the row.

    Parameters
    ----------
    df : Dataframe to be aggregated. Assumed to contain a row for each trade with columns 'price' and 'quantity'.

    Returns
    -------
    Pandas series with the aggregated values
    """
    # Nothing to aggregate if the dataframe is empty or there are empty trades
    volume = df['quantity'].sum()
    if volume == 0:
        return pd.Series({'open': np.nan, 'close': np.nan, 'high': np.nan,
                          'low': np.nan, 'vwap': np.nan, 'vwps': np.nan,
                          'quantity': 0})

    closing = df['price'].iloc[-1]
    opening = df['price'].iloc[0]
    high = df['price'].max()
    low = df['price'].min()
    vwap = (df['price'] * df['quantity']).sum() / volume

    # Computing vwps
    deviation_sum = ((df['price'] - vwap) ** 2 * df['quantity']).sum()
    divisor = max(volume - 1, 1)  # Using N-1 for sample standard deviation and ensuring no division by zero
    vwps = np.sqrt(deviation_sum / divisor)

    return pd.Series({
        'open': opening,
        'close': closing,
        'high': high,
        'low': low,
        'vwap': vwap,
        'vwps': vwps,
        'quantity': volume,
    })

def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function fixes the missing values in the dataframe. It fills the missing values with the last known value.
    The first row is filled with the first value of the dataframe.

    Parameters
    ----------
    df : Dataframe to be fixed. Assumed to have columns 'price' and 'quantity'.

    Returns
    -------
    Dataframe with missing values filled.
    """
    # Forward fill the VWAP values
    df['vwap'] = df['vwap'].ffill()

    # Open, high, low and close are filled to vwap values
    df['open'] = df['open'].fillna(df['vwap'])
    df['high'] = df['high'].fillna(df['vwap'])
    df['low'] = df['low'].fillna(df['vwap'])
    df['close'] = df['close'].fillna(df['vwap'])

    # Quantity is zero since it is not a price
    df['quantity'] = df['quantity'].fillna(0)

    # VWPS is also set to zero as we have no spread
    df['vwps'] = df['vwps'].fillna(0)

    return df

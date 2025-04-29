from typing import Hashable

from pandas import IntervalIndex

from DataProcessing.Processors import FileProcessor
import pandas as pd
import numpy as np
import datetime
import multiprocessing


class LimitOrderBookProcessor(FileProcessor):
    """
    This class processes limit order book data files. It inherits from the FileProcessor class and implements the
    process_file method.
    """
    def __init__(self, file_path: str):
        """
        Initialize the LimitOrderBookProcessor with the path to the file to be processed.
        The file is loaded and the 't' column is converted to datetime format.

        Parameters
        ----------
        file_path : Path to the file to be processed.
        """

        self.aggregated_data = None
        super().__init__(file_path)

    def process_file(self):
        """
        Process the trade data file. This method is called during initialization.
        """
        self.aggregated_data = process_all_days(self.df)

    def get_time_bins(self, date: datetime.date) -> IntervalIndex | None:
        """
        Get the time bins for the given date. This is used to create the time index for the dataframe.

        Parameters
        ----------
        date : Date of the data to be processed.

        Returns
        -------
        DataFrame with the time bins.
        """
        if date not in self.aggregated_data:
            return None

        # Get the data for the given date
        df = self.aggregated_data[date]

        # Add one minute before the first timestamp
        timestamps = pd.Series(df.index)
        first_timestamp = timestamps.iloc[0] - pd.Timedelta(minutes=1)

        # Prepend and rebuild as a pandas DatetimeIndex
        timestamps = pd.concat([
            pd.Series([first_timestamp]),
            timestamps
        ]).reset_index(drop=True)

        # Now build IntervalIndex
        intervals = pd.IntervalIndex.from_arrays(
            timestamps[:-1], timestamps[1:], closed='left'
        )

        return intervals


def empty_rows(df: pd.DataFrame):
    """
    Get the rows that are empty (zeros or NaN). These rows are assumed to be the same as the previous level. These
    rows must be forward filled with the previous row

    Parameters
    ----------
    df : DataFrame to check.

    Returns
    -------

    """
    is_empty = df.isna() | (df == 0)
    missing_indices = is_empty.all(axis=1)

    return missing_indices

def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing data in the dataframe. If the whole row is empty (zeros or NaN), it is assumed that the LOB didn't
    change, hence it filled by the last known value. If the row is partially empty, the previous level is used to fill
    the missing values. In this case, the price is assumed to be the same as the previous level, and the quantity is
    assumed to be zero.

    Parameters
    ----------
    df : DataFrame to fix.

    Returns
    -------
    DataFrame with fixed data.
    """
    # Finding the rows that are completely empty
    missing_indices = empty_rows(df)

    # Forward fill the missing values
    df.loc[missing_indices] = df.loc[missing_indices].ffill()

    # Bid and ask prices that are zero are replaced with NaN and forward filled
    df[['bb1', 'ba1']] = df[['bb1', 'ba1']].replace(0, np.nan)
    df[['bb1', 'ba1']] = df[['bb1', 'ba1']].ffill()

    # Fixing rows that are partially empty. Assumed to have 5 LOB levels
    levels = 5
    for i in range(2, levels + 1):
        # Current level
        bid_price = f'bb{i}'
        bid_quantity = f'bbvol{i}'
        ask_price = f'ba{i}'
        ask_quantity = f'bavol{i}'

        # Previous level
        prev_bid_price = f'bb{i-1}'
        prev_ask_price = f'ba{i-1}'

        # May be zero or nan, hence we put everything to nan
        df[bid_price] = df[bid_price].replace(0, np.nan)
        df[bid_quantity] = df[bid_quantity].replace(0, np.nan)
        df[ask_price] = df[ask_price].replace(0, np.nan)
        df[ask_quantity] = df[ask_quantity].replace(0, np.nan)

        # Price is same as previous level
        df[bid_price] = df[bid_price].fillna(df[prev_bid_price])
        df[ask_price] = df[ask_price].fillna(df[prev_ask_price])

        # Quantity is zero
        df[bid_quantity] = df[bid_quantity].fillna(0)
        df[ask_quantity] = df[ask_quantity].fillna(0)

    return df

def process_day(df: pd.DataFrame, date: datetime.date) -> tuple[datetime.date, pd.DataFrame]:
    """
    Process a single day's data and aggregate it.

    Parameters
    ----------
    df : Dataframe with the data to be processed.
    date: Date of the data to be processed.

    Returns
    -------
    Dataframe with the aggregated data for the day.
    """
    # Fix all the missing data
    df = fix_missing(df)

    # Empty values likely mean that day does not have any data
    if df.isna().any().any():
        return date, pd.DataFrame()

    # Check for zeros in any of the bid and ask prices
    for i in range(1, 6):
        bid_price = f'bb{i}'
        ask_price = f'ba{i}'

        if df[bid_price].eq(0).any() or df[ask_price].eq(0).any():
            return date, pd.DataFrame()

    # Set the time column as the index
    df = df.set_index('t')

    # # Removes the day from the index and converts it to a string
    # df.index = df.index.strftime('%H:%M:%S')

    return date, df

def unpack_args(args) -> tuple[datetime.date, pd.DataFrame]:
    """
    Unpack the arguments for the process_day function. This is used for multiprocessing.

    Parameters
    ----------
    args : Tuple containing the arguments for the process_day function.
    """
    return process_day(*args)

def process_all_days(df: pd.DataFrame) -> dict[Hashable | datetime.date, pd.DataFrame]:
    """
    Process all the trades of a given dataframe. All trades within this day are aggregated to a given interval.
    The opening and closing auctions are also included in the aggregated data, with the first row being the opening auction
    and the last row being the closing auction.

    Parameters
    ----------
    df : DataFrame containing the trade data.

    Returns
    -------
    Dictionary with dates as keys and DataFrames as values. Each DataFrame contains the aggregated trade data for that date.
    """

    if df.empty:
        return {}

    # The nature column is not needed for the processing
    df = df.drop(columns=['nature'])

    # Group into days
    groups = df.groupby(df['t'].dt.date)

    n_cores = multiprocessing.cpu_count()

    # Use multiprocessing to process each group in parallel
    items = [(group, date) for date, group in groups]
    with multiprocessing.Pool(n_cores) as pool:
        processed_groups = pool.map(unpack_args, items)

    # Concatenate the results into a dictionary
    processed_groups = {date: data for date, data in processed_groups if not data.empty}

    return processed_groups
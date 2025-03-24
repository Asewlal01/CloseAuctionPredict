from typing import List

import numpy as np
import pandas as pd
import gzip
import datetime
import os
import multiprocessing

from pandas import Timestamp
from tqdm import tqdm

class TradeSplitter:
    def __init__(self, stock_info_path, exchange_times_path, save_path, n_cores: int=-1):
        """
        Initialize the Trades object.

        Parameters
        ----------
        stock_info_path : Path to file which has list of stocks with their corresponding traded exchanges.
        exchange_times_path : Path to file which has opening and closing times of continuous trading for each exchange.
        n_cores : Number of cores to use for processing. Default is -1, which uses all available cores.
        """
        self.stocks = pd.read_csv(stock_info_path)
        self.exchange_times = pd.read_csv(exchange_times_path, keep_default_na=False)
        self.save_path = save_path
        self.n_cores = n_cores

    def process_all(self, trade_path: str) -> None:
        """
        Process all trade data files in the specified directory.

        Parameters
        ----------
        trade_path : Path to the directory containing the trade data files.

        Returns
        -------
        None
        """
        # Get all files in the directory
        files = os.listdir(trade_path)
        items = [
            (os.path.join(trade_path, file), self.stocks, self.exchange_times, self.save_path) for file in files
        ]

        with multiprocessing.Pool(self.n_cores) as pool:
            _ = list(
                tqdm(pool.imap(unpack_args, items), total=len(items))
            )

def unpack_args(args):
    return process_file(*args)

def load_from_gzip(zip_file: str) -> pd.DataFrame:
    """
    Load a zipped trade data file. The returned dataframe has the following columns:
    - time: The time of the trade.
    - price: The price of the trade.
    - quantity: The quantity of the trade.
    - flag: The flag indicating the type of trade (e.g., AUCTION, NORMAL).

    Parameters
    ----------
    zip_file : Path to the zipped trade data file.

    Returns
    -------
    Dataframe containing the trade data.
    """
    with gzip.open(zip_file, 'rt') as f:
        df = pd.read_csv(f, header=0, names=['t', 'price', 'quantity', 'flag'], engine='pyarrow')
    return df

def get_name_and_exchange(zip_file: str, stock_info) -> tuple[str, str]:
    """
    Get the exchange and name of the stock from the zipped file name. The name is in format:
    trade_data_<begin_date>_<end_date>_<id>.gz

    Parameters
    ----------
    zip_file : File name of the zipped trade data file.
    stock_info : Dataframe containing stock information.

    Returns
    -------
    Name of stock and exchange it is traded on.
    """
    # Converts trade_data_<begin_date>_<end_date>_<id>.gz to <id>.gz
    id_with_extension = zip_file.split('_')[-1]
    # Converts <id>.gz to <id>
    id = id_with_extension.split('.')[0]
    id = int(id)

    stock_row = stock_info[stock_info['syd'] == id]
    if stock_row.empty:
        raise ValueError(f"Stock with id {id} not found in stock information.")

    bb = stock_row['bb'].values[0]
    name, exchange, _ = bb.split(' ')

    return name, exchange

def get_opening_and_closing_times(exchange: str, exchange_times: pd.DataFrame) -> tuple[str, str, str]:
    """
    Get the opening and closing times of continuous trading for the given exchange.

    Parameters
    ----------
    exchange : Name of the exchange.
    exchange_times : Dataframe containing opening and closing times of continuous trading for each exchange.

    Returns
    -------
    Tuple containing the opening and closing times.
    """
    row = exchange_times[exchange_times['bb'] == exchange]
    if row.empty:
        raise ValueError(f"Exchange {exchange} not found in exchange times.")

    opening_time = row['opening'].values[0]
    closing_time = row['closing'].values[0]
    timezone = row['timezones'].values[0]

    return opening_time, closing_time, timezone

def convert_to_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'time' column to a datetime object.

    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    Dataframe with the 'time' column converted to a datetime object.
    """
    df['t'] = pd.to_datetime(df['t'], format='mixed')
    return df

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows which are not normal or auction trades.

    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    Dataframe with only normal and auction trades.
    """
    return df[
        df['flag'].isin(['AUCTION', 'NORMAL'])
    ]

def merge_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows which occurred at the same time and have the same trading price.

    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    Dataframe with combined rows.
    """
    return df.groupby(['t', 'price'], as_index=False).agg(
        {
            'quantity': 'sum',
            'flag': 'first'
        }
    ).reset_index(drop=True)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the trade data dataframe. This function calls all the other processing functions in order.

    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    Dataframe with processed trade data.
    """
    df = convert_to_time(df)
    df = filter_rows(df)
    df = merge_rows(df)

    return df

def convert_to_timestamp(date: datetime.date, opening_time: str, closing_time: str,
                         timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Convert the opening and closing times to a timestamp object by attaching the date to the time. The timestamp is
    also converted to UTC by using the timezone information.

    Parameters
    ----------
    date : Datetime object to be converted.
    opening_time : Opening time of continuous trading for the exchange.
    closing_time : Closing time of continuous trading for the exchange.
    timezone : Timezone of the exchange.

    Returns
    -------
    Timestamp object.
    """
    opening_timestamp = pd.Timestamp(f'{date} {opening_time}', tz=timezone)
    closing_timestamp = pd.Timestamp(f'{date} {closing_time}', tz=timezone)

    # Convert to UTC
    opening_timestamp = opening_timestamp.tz_convert('UTC')
    closing_timestamp = closing_timestamp.tz_convert('UTC')

    return opening_timestamp, closing_timestamp

def filter_normal_trades(df: pd.DataFrame, opening_time: pd.Timestamp, closing_time: pd.Timestamp) -> pd.DataFrame:
    """
    Filter out normal trades that are before the opening time or after the closing time.

    Parameters
    ----------
    df : Dataframe containing the trade data.
    opening_time : Opening time of continuous trading for the exchange.
    closing_time : Closing time of continuous trading for the exchange.

    Returns
    -------
    Dataframe with auction trades and filtered normal trades.
    """
    return df[
        (df['flag'] == 'AUCTION') |
        ((df['t'] >= opening_time) & (df['t'] <= closing_time))
    ].reset_index(drop=True)

def has_auction_and_normal(df: pd.DataFrame) -> bool:
    """
    Check if the dataframe contains both auction and normal trades.

    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    bool : True if the dataframe contains both auction and normal trades, False otherwise.
    """

    if df.empty:
        return False

    # Check if there is at least one auction row
    has_auction = (df['flag'] == 'AUCTION').any()
    # Check if there is at least one normal row
    has_normal = (df['flag'] == 'NORMAL').any()

    return has_auction and has_normal


def has_opening_and_closing(df: pd.DataFrame, closing_timestamp: pd.Timestamp) -> bool:
    """
    Check if the dataframe has an opening and closing auction trade.
    The first and last rows of the dataframe should be auction trades as these are the first and last trades of the day
    respectively.

    Parameters
    ----------
    df : Dataframe containing the trade data.
    closing_timestamp : Closing time of continuous trading for the exchange.

    Returns
    -------
    bool : True if the dataframe contains auction trades, False otherwise.
    """
    # The first and last rows of the dataframe should be auction trades
    first_trade = df.iloc[0]
    last_trade = df.iloc[-1]

    first_is_auction = first_trade['flag'] == 'AUCTION'
    last_is_auction = last_trade['flag'] == 'AUCTION'

    # The last trade should be after the closing time
    last_trade_after_closing = last_trade['t'] > closing_timestamp

    return first_is_auction and last_is_auction and last_trade_after_closing

def create_empty_row(timestamp: pd.Timestamp) -> list[Timestamp | float | float | str]:
    """
    Create an empty row with the given timestamp. This row will contain the following columns:
    - t: The timestamp of the trade.
    - price: The price of the trade (Nan).
    - quantity: The quantity of the trade (0).
    - flag: The flag indicating the type of trade (NORMAL).

    Parameters
    ----------
    timestamp : Timestamp for the empty row.

    Returns
    -------
    Dataframe containing the empty row.
    """
    return [timestamp, np.nan, 0.0, 'NORMAL']

def add_empty_trades(df: pd.DataFrame, opening_timestamp: pd.Timestamp, closing_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Add empty trades to the dataframe just after the opening and just before the closing time.
    This is done to ensure that when aggregating the dataset, aggregation starts from the opening time
    and ends at the closing time.

    Parameters
    ----------
    df : Dataframe containing the trade data.
    opening_timestamp : Opening time of continuous trading for the exchange.
    closing_timestamp : Closing time of continuous trading for the exchange.

    Returns
    -------
    Dataframe with empty trades added.
    """
    # We need the normal trades to find which indices and timestamps to use
    normal_trades = df[df['flag'] == 'NORMAL']

    # The empty opening rows is placed just before the first normal trade
    opening_index = normal_trades.index[0] - 0.5
    opening_timestamp += pd.Timedelta(microseconds=1)

    # The empty closing rows is placed just after the last normal trade
    closing_index = normal_trades.index[-1] + 0.5
    closing_timestamp -= pd.Timedelta(microseconds=1)

    opening_row = create_empty_row(opening_timestamp)
    closing_row = create_empty_row(closing_timestamp)

    # We use 0.5 to make sure that we do not overwrite the existing rows, but still are between normal and auction rows
    df.loc[opening_index] = opening_row
    df.loc[closing_index] = closing_row

    return df.sort_index()

def process_by_group(df: pd.DataFrame, opening_time: str, closing_time: str, timezone, save_path: str) -> None:
    """
    Process the grouped data of a given stock. This grouping is based on the day of the trade.

    Parameters
    ----------
    df : Dataframe containing the trade data.
    opening_time : Opening time of continuous trading for the exchange.
    closing_time : Closing time of continuous trading for the exchange.
    timezone : Timezone of the exchange.
    save_path : Path to save the processed data.

    Returns
    -------
    None
    """
    date_groups = df.groupby(df['t'].dt.date)

    for date, group in date_groups:
        opening_timestamp, closing_timestamp = convert_to_timestamp(date, opening_time, closing_time, timezone)
        filtered_group = filter_normal_trades(group, opening_timestamp, closing_timestamp)

        if not has_auction_and_normal(filtered_group):
            continue

        if has_opening_and_closing(filtered_group, closing_timestamp):
            filtered_group = add_empty_trades(filtered_group, opening_timestamp, closing_timestamp)
            filtered_group.to_parquet(f'{save_path}/{date}.parquet', engine="pyarrow", index=False)

def process_file(file: str, stock_info: pd.DataFrame, exchange_times: pd.DataFrame, save_path: str) -> None:
    """
    Process a trade data file.

    Parameters
    ----------
    file : Path to the trade data file.
    stock_info : Dataframe containing stock information.
    exchange_times : Dataframe containing opening and closing times of continuous trading for each exchange.
    save_path : Path to save the processed data.

    Returns
    -------
    None
    """

    # Loading
    df = load_from_gzip(file)
    name, exchange = get_name_and_exchange(file, stock_info)
    opening_time, closing_time, timezone = get_opening_and_closing_times(exchange, exchange_times)

    # Global processing
    df = process_dataframe(df)

    # Processing per day and saving
    save_path = f'{save_path}/{name}'
    os.makedirs(save_path, exist_ok=True)
    process_by_group(df, opening_time, closing_time, timezone, save_path)





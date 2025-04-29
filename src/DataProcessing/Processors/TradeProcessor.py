from typing import Hashable
from DataProcessing.Processors import FileProcessor
from DataProcessing.Processors import LimitOrderBookProcessor
import pandas as pd
import numpy as np
import datetime
import multiprocessing

class TradeProcessor(FileProcessor):
    """
    This class processes trade data files. It inherits from the FileProcessor class and implements the process_file method.
    The trade data is assumed to be a gzip compressed CSV file.
    """
    def __init__(self, file_path: str, lob_processor: LimitOrderBookProcessor):
        """
        Initialize the TradeProcessor with the path to the file to be processed and the interval for aggregation.
        The file is loaded and the 't' column is converted to datetime format.

        Parameters
        ----------
        file_path : Path to the file to be processed.
        lob_processor : Processor of limit order book data.
        """
        self.aggregated_data = None
        self.lob_processor = lob_processor
        super().__init__(file_path)

    def process_file(self):
        """
        Process the trade data file. This method is called during initialization.
        """
        self.aggregated_data = process_all_days(self.df, self.lob_processor)

def has_auctions(df: pd.DataFrame) -> bool:
    """
    Check if the dataframe has auction data.

    Parameters
    ----------
    df : DataFrame to check.

    Returns
    -------
    bool : True if the dataframe has auction data, False otherwise.
    """
    auctions = df[df['flag'] == 'AUCTION']
    if auctions.empty:
        return False

    # Even if not empty, we may only have one auction
    if len(auctions) < 2:
        return False

    # Check if the auctions are consecutive
    max_idx_diff = auctions.index.diff().max()
    if max_idx_diff == 1:
        return False

    return True

def has_data(df: pd.DataFrame) -> bool:
    """
    Check if the dataframe has auction and trade data.

    Parameters
    ----------
    df : DataFrame to check.

    Returns
    -------
    bool : True if the dataframe has data, False otherwise.
    """
    trades = df[df['flag'] == 'NORMAL']
    if trades.empty or not has_auctions(df):
        return False

    return True

def merge_auction_data(auction_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all auction rows into a single dataframe with one row

    Parameters
    ----------
    auction_rows : DataFrame containing the auction data.

    Returns
    -------
    Series containing the merged auction data.
    """
    price = auction_rows['price'].iloc[0]
    return pd.DataFrame({
        'open': [price],
        'close': [price],
        'high': [price],
        'low': [price],
        'vwap': [price],
        'quantity': [auction_rows['quantity'].sum()],
    })

def get_auction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the auction data from the trade data.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    Returns
    -------
    DataFrame containing the aggregated trade data.
    """
    auction_df = df[df['flag'] == 'AUCTION']

    # Same auctions have a difference of 1 in their indices
    idx = auction_df.index.diff().argmax()
    opening_auction = auction_df.iloc[:idx]
    closing_auction = auction_df.iloc[idx:]

    opening_merged = merge_auction_data(opening_auction)
    closing_merged = merge_auction_data(closing_auction)

    # Reset index removes the normal index and replaces it with opening and closing
    return pd.concat([opening_merged, closing_merged],
                     keys=['opening', 'closing']).reset_index(level=1, drop=True)

def aggregate_interval(df: pd.DataFrame) -> pd.Series:
    """
    This function aggregates the trades and quantities within a dataframe to a single row. This row contains the
    open, high, low, close and vwap prices as well as the total volume of trades.

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
                          'low': np.nan, 'vwap': np.nan, 'quantity': 0})

    closing = df['price'].iloc[-1]
    opening = df['price'].iloc[0]
    high = df['price'].max()
    low = df['price'].min()
    vwap = (df['price'] * df['quantity']).sum() / volume

    return pd.Series({
        'open': opening,
        'close': closing,
        'high': high,
        'low': low,
        'vwap': vwap,
        'quantity': volume,
    })

def aggregate_trades(df: pd.DataFrame, lob_processor: LimitOrderBookProcessor, date: datetime.date) -> pd.DataFrame:
    """
    This function aggregates all the trades of a given df using a resampling interval.

    Parameters
    ----------
    df : Dataframe to be aggregated. Assumed to have columns 'price' and 'quantity'.
    lob_processor : Processor of limit order book data.
    date : Date of the trades to be processed.

    Returns
    -------
    Dataframe of resampled trade data
    """

    # Get the time bins
    time_bins = lob_processor.get_time_bins(date)
    if time_bins is None:
        return pd.DataFrame()

    trade_bins = pd.cut(df['t'], bins=time_bins)

    # Group the trades by the time bins
    groups = df.groupby(trade_bins, observed=True)
    aggregated_trades = groups.apply(aggregate_interval, include_groups=False)

    if aggregated_trades.empty:
        return pd.DataFrame()

    intervals = pd.IntervalIndex(aggregated_trades.index)
    aggregated_trades.index = intervals.right

    # Check if unique
    if not aggregated_trades.index.is_unique:
        raise ValueError("Timestanp indices have non-unique indices")

    return aggregated_trades

def process_day(df: pd.DataFrame, lob_processor: LimitOrderBookProcessor, date: datetime.date) -> tuple[datetime.date, pd.DataFrame]:
    """
    Process all the trades of a given day. All trades within this day are aggregated to a given interval.
    The opening and closing auctions are also included in the aggregated data, with the first row being the opening auction
    and the last row being the closing auction.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    lob_processor : Processor of limit order book data.
    date : Date of the trades to be processed.

    Returns
    -------
    DataFrame containing the aggregated trade data.
    """
    if not has_data(df):
        return date, pd.DataFrame()

    # Get auction data
    auction_data = get_auction_data(df)
    if auction_data.empty:
        return date, pd.DataFrame()

    # Aggregation only uses normal trades
    df = df[df['flag'] == 'NORMAL']
    aggregated_trades = aggregate_trades(df, lob_processor, date)
    if aggregated_trades.empty:
        return date, pd.DataFrame()

    # Removing the day from the index and converting the time to a string
    aggregated_trades.index = aggregated_trades.index.strftime('%H:%M:%S:%f')

    # Concatenate the aggregated trades with the auction data
    aggregated_trades = pd.concat([aggregated_trades, auction_data])

    return date, aggregated_trades

def unpack_args(args) -> tuple[datetime.date, pd.DataFrame]:
    """
    Unpack the arguments for the process_day function. This is used for multiprocessing.

    Parameters
    ----------
    args : Tuple containing the arguments for the process_day function.
    """
    return process_day(*args)

def process_all_days(df: pd.DataFrame, lob_processor: LimitOrderBookProcessor) -> dict[Hashable | datetime.date, pd.DataFrame]:
    """
    Process all the trades of a given dataframe. All trades within this day are aggregated to a given interval.
    The opening and closing auctions are also included in the aggregated data, with the first row being the opening auction
    and the last row being the closing auction.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    lob_processor : Processor of limit order book data.

    Returns
    -------
    Dictionary with dates as keys and DataFrames as values. Each DataFrame contains the aggregated trade data for that date.
    """
    # Group into days
    groups = df.groupby(df['t'].dt.date)

    n_cores = multiprocessing.cpu_count()

    # Use multiprocessing to process each group in parallel
    items = [(group, lob_processor, date) for date, group in groups]
    with multiprocessing.Pool(n_cores) as pool:
        processed_groups = pool.map(unpack_args, items)

    # Concatenate the results into a dictionary
    processed_groups = {date: data for date, data in processed_groups if not data.empty}

    return processed_groups
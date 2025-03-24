import pandas as pd
import os
import multiprocessing
import numpy as np
from tqdm import tqdm


class TradeAggregator:
    """
    Class to aggregate all trades over a day to a given interval.
    """
    def __init__(self, save_dir: str, interval: str, n_cores: int = None):
        """
        Initialize the TradeAggregator object.

        Parameters
        ----------
        save_dir : Directory to save the aggregated trade data files.
        interval : Interval to aggregate the trade data to.
        n_cores : Number of cores to use for multiprocessing. Default is None, which uses all available cores
        """
        self.save_dir = save_dir
        self.interval = interval
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()

        if not os.path.exists(self.save_dir):
            print(f'Creating directory {self.save_dir}')
            os.makedirs(self.save_dir)

    def aggregate_all(self, trade_path: str) -> None:
        """
        Aggregate all trades for each stock within trade_path to the given interval.

        Parameters
        ----------
        trade_path : Directory containing the trade data for each stock. This directory should contain subdirectories
        for each stock, which contain the trade data files as parquet files.

        Returns
        -------
        None
        """

        # Get all stock
        stocks = os.listdir(trade_path)
        stocks.sort()

        items = [(os.path.join(trade_path, stock), self.interval, os.path.join(self.save_dir, stock)) for stock in stocks]
        with multiprocessing.Pool(self.n_cores) as pool:
            _ = list(
                tqdm(pool.imap(unpack_args, items), total=len(items))
            )

def unpack_args(args):
    process_stock(*args)

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
    return pd.DataFrame({
        'price': [auction_rows['price'].mean()],
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
    This function aggregates the trades and quantities within a dataframe to a single row. This row contains
    the volume weighted average price (VWAP) and the total quantity traded.

    Parameters
    ----------
    df : Dataframe to be aggregated. Assumed to contain a row for each trade with columns 'price' and 'quantity'.

    Returns
    -------
    Pandas series with the aggregated values
    """
    # Nothing to aggregate if the dataframe is empty or there are empty trades
    if df['quantity'].sum() == 0:
        return pd.Series({'price': np.nan, 'quantity': 0})

    volume = df['quantity'].sum()
    vwap = (df['price'] * df['quantity']).sum() / volume

    return pd.Series({'price': vwap, 'quantity': volume})

def aggregate_trades(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    This function aggregates all the trades of a given df using a resampling interval.

    Parameters
    ----------
    df : Dataframe to be aggregated. Assumed to have columns 'price' and 'quantity'.
    interval : Aggregation interval using a format that is supported by Pandas resample function.

    Returns
    -------
    Dataframe of resampled trade data
    """
    resample = df.resample(interval, on='t', label='left')

    return resample.apply(aggregate_interval)

def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    When there is no trade within the given interval, the price is set to nan. This function sets those function to
    the last known price. Missing values for the volume are set to 0.

    Parameters
    ----------
    df : Dataframe of the aggregated trade data

    Returns
    -------
    Dataframe with missing values filled in
    """
    # No trades means no change in price, hence vwap did not change
    df['price'] = df['price'].ffill()

    # No trades mean zero volume
    df['quantity'] = df['quantity'].fillna(0)

    return df

def process_trade_file(trade_file: str, interval: str) -> pd.DataFrame:
    """
    Process a single trade file to aggregate the trades to the given interval.

    Parameters
    ----------
    trade_file : Path to the trade file.
    interval : Interval to aggregate the trade data to.

    Returns
    -------
    DataFrame containing the aggregated trade data.
    """
    df = pd.read_parquet(trade_file)

    # Get auction data
    auction_data = get_auction_data(df)

    # Aggregation only uses normal trades
    df = df[df['flag'] == 'NORMAL']
    aggregated_trades = aggregate_trades(df, interval)

    # Removing the day from the index and converting the time to a string
    aggregated_trades.index = aggregated_trades.index.strftime('%H:%M:%S')
    indices = aggregated_trades.index.tolist()
    indices = ['opening'] + indices + ['closing']

    # Concatenate the aggregated trades with the auction data
    aggregated_trades = pd.concat([aggregated_trades, auction_data])
    aggregated_trades = aggregated_trades.loc[indices]

    # Fix missing values is called after merging to ensure that opening price can also be used
    aggregated_trades = fix_missing(aggregated_trades)

    return aggregated_trades

def process_stock(stock_path: str, interval: str, save_path: str) -> None:
    """
    Process all the trade files for a single stock to aggregate the trades to the given interval. All aggregated trades
    are saved to the save path with the date as the filename.

    Parameters
    ----------
    stock_path : Path to the stock directory containing the trade files.
    interval : Aggregation interval using a format that is supported by Pandas resample function.
    save_path : Path to save the aggregated trade data.

    Returns
    -------
    None
    """
    dates = os.listdir(stock_path)
    dates.sort()

    os.makedirs(save_path, exist_ok=True)
    for date in dates:
        date_path = os.path.join(stock_path, date)
        aggregated_df = process_trade_file(date_path, interval)
        aggregated_df.to_parquet(os.path.join(save_path, date), index=True)


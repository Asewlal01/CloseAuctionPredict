import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
from DataProcessing import TradeSplitter, TradeAggregator

class TradePreprocessor:
    """
    A class to preprocess trade data for analysis. This class can be seen as a combined version of the TradeSplitter
    and TradeAggregator classes, providing a unified interface for splitting and aggregating trade data without the need
    for saving intermediate splitted data to disk.
    """
    def __init__(self, save_dir: str, stock_info_path: str, exchange_times_path: str,
                 interval: str, n_cores: int = None):
        """
        Initialize the TradePreprocessor object.

        Parameters
        ----------
        save_dir : Directory to save the aggregated trade data files.
        stock_info_path : Path to the stock information file.
        exchange_times_path : Path to the exchange times file.
        interval : Interval to aggregate the trade data to.
        n_cores : Number of cores to use for multiprocessing. Default is None, which uses all available cores.
        """
        # These are related to splitting
        self.stock_info = pd.read_csv(stock_info_path)
        self.exchange_times = pd.read_csv(exchange_times_path, keep_default_na=False)

        # Needed for aggregating
        self.interval = interval

        self.save_dir = save_dir
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()

        if not os.path.exists(self.save_dir):
            print(f'Creating directory {self.save_dir}')
            os.makedirs(self.save_dir)

    def process_all(self, trade_path: str) -> None:
        """
        Process all trades for each stock within trade_path to the given interval.

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

        items = [(os.path.join(trade_path, stock), self.stock_info, self.exchange_times,
                  self.interval, self.save_dir) for stock in stocks]

        with multiprocessing.Pool(self.n_cores) as pool:
            for _ in tqdm(pool.imap(unpack_args, items), total=len(items)):
                pass

def unpack_args(args):
    """
    Unpack the arguments for processing a single stock.

    Parameters
    ----------
    args : tuple
        A tuple containing the arguments for processing a single stock.

    Returns
    -------
    None
    """
    process_stock(*args)

def process_stock(file, stock_info: pd.DataFrame, exchange_times: pd.DataFrame,
                  interval: str, save_path: str) -> None:
    """
    Process a single stock's trade data.

    Parameters
    ----------
    file : File path to the trade data for the stock.
    stock_info : DataFrame containing stock information.
    exchange_times : DataFrame containing exchange times.
    interval : Interval to aggregate the trade data to.
    save_path : Path to save the aggregated trade data.

    Returns
    -------
    None
    """

    # Load and split the data
    name, groups = split_to_groups(file, stock_info, exchange_times)

    # Making the directory if it does not exist
    save_path = os.path.join(save_path, name)
    os.makedirs(save_path, exist_ok=True)

    # Aggregating the data
    for date, group in groups.items():
        # Aggregating the group
        aggregated_group = aggregate_group(group, interval)

        # Saving the aggregated data
        save_file = os.path.join(save_path, f'{date}.parquet')
        aggregated_group.to_parquet(save_file)

def aggregate_group(group: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Aggregate the trade data to the given interval.

    Parameters
    ----------
    group : DataFrame containing the trade data for a single day.
    interval : Interval to aggregate the trade data to.

    Returns
    -------
    DataFrame containing the aggregated trade data.
    """

    # Get auction data
    auction_data = TradeAggregator.get_auction_data(group)

    # Aggregation only uses normal trades
    group = group[group['flag'] == 'NORMAL']
    aggregated_trades = TradeAggregator.aggregate_trades(group, interval)

    # Removing the day from the index and converting the time to a string
    aggregated_trades.index = aggregated_trades.index.strftime('%H:%M:%S')
    indices = aggregated_trades.index.tolist()
    indices = ['opening'] + indices + ['closing']

    # Concatenate the aggregated trades with the auction data
    aggregated_trades = pd.concat([aggregated_trades, auction_data])
    aggregated_trades = aggregated_trades.loc[indices]

    # Fix missing values is called after merging to ensure that opening price can also be used
    aggregated_trades = TradeAggregator.fix_missing(aggregated_trades)

    return aggregated_trades

def split_to_groups(file: str, stock_info: pd.DataFrame, exchange_times: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Load the stock data, and group it by days

    Parameters
    ----------
    file : File path to the trade data for the stock.
    stock_info : DataFrame containing stock information.
    exchange_times : DataFrame containing exchange times.

    Returns
    -------
    Dictionary of DataFrames, where the keys are the dates and the values are the DataFrames for each date.
    """
    # Loading and processing
    df = TradeSplitter.load_from_gzip(file)
    df = TradeSplitter.process_dataframe(df)

    # Get the stock name and exchange
    name, exchange = TradeSplitter.get_name_and_exchange(file, stock_info)
    opening_time, closing_time, timezone = TradeSplitter.get_opening_and_closing_times(exchange, exchange_times)

    # Grouping by the date
    date_groups = df.groupby(df['t'].dt.date)

    groups_to_keep = {}
    for date, group in date_groups:
        opening_timestamp, closing_timestamp = TradeSplitter.convert_to_timestamp(date, opening_time, closing_time, timezone)
        filtered_group = TradeSplitter.filter_normal_trades(group, opening_timestamp, closing_timestamp)

        if not TradeSplitter.has_auction_and_normal(filtered_group):
            continue

        if TradeSplitter.has_opening_and_closing(filtered_group, closing_timestamp):
            date = date.strftime('%Y-%m-%d')
            groups_to_keep[date] = filtered_group

    return name, groups_to_keep








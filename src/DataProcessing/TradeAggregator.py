import pandas as pd
import os
import multiprocessing
import numpy as np


class TradeAggregator:
    """
    Class to aggregate all trades over a day to a given interval.
    """
    def __init__(self, save_dir: str, exchange_info: pd.Series, interval: str, n_cores: int = None):
        """
        Initialize the TradeAggregator object.

        Parameters
        ----------
        save_dir : Directory to save the aggregated trade data files.
        interval : Interval to aggregate the trade data to.
        n_cores : Number of cores to use for multiprocessing. Default is None, which uses all available cores
        """
        self.save_dir = save_dir
        self.exchange_info = exchange_info
        self.interval = interval
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()

        if not os.path.exists(self.save_dir):
            print(f'Creating directory {self.save_dir}')
            os.makedirs(self.save_dir)

        self.opening = None
        self.closing = None
        self.exchange_times()

    def exchange_times(self):
        """
        This function sets the opening and closing time of the exchange based on the exchange row.

        Returns
        -------
        Tuple with the opening and closing time of the exchange
        """
        # Get opening and closing time
        opening = self.exchange_info['opening'].values[0]
        opening = pd.to_datetime(opening, format='%H:%M:%S').time()
        self.opening = opening

        closing = self.exchange_info['closing'].values[0]
        closing = pd.to_datetime(closing, format='%H:%M:%S').time()
        self.closing = closing

    def aggregate_stock(self, stock_path: str) -> None:
        """
        Aggregate all trades over a day to a given interval for a single stock.

        Parameters
        ----------
        stock_path : Path to the stock file to aggregate

        Returns
        -------
        None
        """

        stock_name = stock_path.split('/')[-1]
        save_dir_stock = os.path.join(self.save_dir, stock_name)
        if not os.path.exists(save_dir_stock):
            os.makedirs(save_dir_stock)

        files = [os.path.join(stock_path, file) for file in os.listdir(stock_path)]
        items = [(file, save_dir_stock, self.opening, self.closing, self.interval) for file in files]

        # No need for pooling if there is only one core
        if self.n_cores == 1:
            print(f'No Pooling for {stock_name}')
            for item in items:
                process_file(*item)
            return

        with multiprocessing.Pool(self.n_cores) as pool:
            pool.starmap(process_file, items)


def time_increment(opening_time, exchange_opening):
    timestamp = pd.Timestamp(opening_time).time()

    delta1 = pd.Timedelta(hours=timestamp.hour, minutes=timestamp.minute, seconds=timestamp.second, microseconds=timestamp.microsecond)
    delta2 = pd.Timedelta(hours=exchange_opening.hour, minutes=exchange_opening.minute, seconds=exchange_opening.second, microseconds=exchange_opening.microsecond)

    time_adjustment = delta2 - delta1

    return time_adjustment


def move_to_opening(df: pd.DataFrame, exchange_open: pd.Timestamp) -> pd.DataFrame:
    """
    This function increments the time of the dataframe to match the opening time. Exchanges denote their opening times in a
    timezone which may be different from the timezone of the data.

    Parameters
    ----------
    df : Dataframe with the trade data
    exchange_open : Opening time of the exchange in the format 'HH:MM:SS'

    Returns
    -------
    Dataframe with the time incremented to the opening time
    """
    if not (df['flag'] == 'AUCTION').any():
        return pd.DataFrame()

    # Incrementing requires opening auction row
    opening_auction_row = df[df['flag'] == 'AUCTION'].iloc[0]
    opening_auction_time = opening_auction_row['time']

    # We want to make sure that the opening time is exactly the same as the exchange opening time
    adjustment = time_increment(opening_auction_time, exchange_open)
    df['time'] = df['time'] + adjustment

    return df

def auction_normal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function splits the trade data into auction and normal trades.

    Parameters
    ----------
    df : Dataframe with the trade data

    Returns
    -------
    Normal and Auction trade dataframes
    """

    auction = df[df['flag'] == 'AUCTION'].copy()
    normal = df[df['flag'] == 'NORMAL'].copy()

    return auction, normal

def get_auction_data(df: pd.DataFrame, exchange_close: pd.Timestamp) -> pd.DataFrame:
    """
    This function returns the auction data from the trade data with only auction trades.

    Parameters
    ----------
    df : Dataframe with the trade data with only auction trades
    exchange_close : Closing time of the exchange in the format 'HH:MM:SS'

    Returns
    -------
    Dataframe with only the auction trades
    """
    # Empty df
    if df.empty:
        return df

    opening = df.iloc[0]
    closing = df.iloc[-1]

    # On normal days the closing auction time is after the closing time
    if closing['time'].time() < exchange_close:
        return pd.DataFrame()

    # If the opening and closing price are the same then one of the prices is missing
    if opening['price'] == closing['price']:
        return pd.DataFrame()

    # Get the auction price and quantity
    opening_price = opening['price']
    opening_rows = df[df['price'] == opening_price]
    opening_quantity = opening_rows['quantity'].sum()

    closing_price = closing['price']
    closing_rows = df[df['price'] == closing_price]
    closing_quantity = closing_rows['quantity'].sum()

    # OCHL and VWAP are the same as the auction price
    df = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'volume'])
    df.loc['opening'] = [opening_price] * 4 + [opening_quantity]
    df.loc['closing'] = [closing_price] * 4 + [closing_quantity]

    return df

def add_exchange_times(df: pd.DataFrame, exchange_open: pd.Timestamp, exchange_close: pd.Timestamp) -> pd.DataFrame:
    """
    This function adds the opening and closing time of the exchange to the dataframe.

    Parameters
    ----------
    df : Dataframe of the trade data with only the normal trades
    exchange_open : Opening time of the exchange in the format 'HH:MM:SS'
    exchange_close : Closing time of the exchange in the format 'HH:MM:SS'

    Returns
    -------
    Dataframe with the opening and closing time of the exchange added
    """

    day = df['time'].dt.date.iloc[0]
    tz = df['time'].dt.tz

    # # Checking if opening time is needed
    # opening_day = pd.Timestamp(f'{day} {exchange_open}', tz=tz)
    # opening_index = df.index[0] - 1
    # first_time = df['time'].iloc[0]
    # if first_time.hour != opening_day.hour or first_time.minute != opening_day.minute:
    #     df.loc[opening_index] = [opening_day, np.nan, np.nan, 'NORMAL']

    # Checking if closing time is needed
    last_time = df['time'].iloc[-1]
    closing_day = pd.Timestamp(f'{day} {exchange_close}', tz=tz) - pd.Timedelta(seconds=1)
    closing_index = df.index[-1] + 1
    if last_time.hour != closing_day.hour or last_time.minute != closing_day.minute:
        df.loc[closing_index] = [closing_day, np.nan, np.nan, 'NORMAL']

    return df.sort_index()

def filter_time(df: pd.DataFrame, exchange_open: pd.Timestamp, exchange_close: pd.Timestamp) -> pd.DataFrame:
    """
    This function removes trades outside the exchange opening and closing times. These trades are considered
    to be erroneous.

    Parameters
    ----------
    df : Dataframe of the trade data with only the normal trades
    exchange_open : Opening time of the exchange in the format 'HH:MM:SS'
    exchange_close : Closing time of the exchange in the format 'HH:MM:SS'

    Returns
    -------
    Dataframe with only the trades within the exchange opening and closing times
    """

    return df[(df['time'].dt.time >= exchange_open) & (df['time'].dt.time <= exchange_close)]

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
    # No point in aggregating if the dataframe is empty
    if df['quantity'].sum() == 0:
        return pd.Series({'open': np.nan, 'close': np.nan, 'high': np.nan, 'low': np.nan, 'volume': np.nan})

    open = df.iloc[0]['price']
    close = df.iloc[-1]['price']
    high = df['price'].max()
    low = df['price'].min()
    volume = df['quantity'].sum()
    # vwap = (df['price'] * df['quantity']).sum() / volume

    return pd.Series({'open': open, 'close': close, 'high': high, 'low': low, 'volume': volume})

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
    resample = df.resample(interval, on='time', label='left')

    return resample.apply(aggregate_interval)

def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function fills in missing values in the aggregated trade data.

    Parameters
    ----------
    df : Dataframe of the aggregated trade data

    Returns
    -------
    Dataframe with missing values filled in
    """
    # The VWAP and close are filled with the previous value
    df['close'] = df['close'].ffill()

    # No trades mean zero volume
    df['volume'] = df['volume'].fillna(0)

    # The open, close, high and low are filled with vwap of the current minute
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])

    return df

def process_df(df: pd.DataFrame, exchange_open: pd.Timestamp, exchange_close: pd.Timestamp,
               interval: str) -> pd.DataFrame:
    """
    This function processes all the trade data for a single day. It returns a df with the resampled data and the auction data.

    Parameters
    ----------
    df : Dataframe with the trade data
    exchange_open : Opening time of the exchange in the format 'HH:MM:SS'
    exchange_close : Closing time of the exchange in the format 'HH:MM:SS'
    interval : Aggregation interval using a format that is supported by Pandas resample function

    Returns
    -------
    Dataframe with the resampled data and the auction data
    """

    df = move_to_opening(df, exchange_open)
    # No auction data
    if df.empty:
        return df

    auction, normal = auction_normal_split(df)
    auction_df = get_auction_data(auction, exchange_close)

    # Miss opening or closing auction
    if auction_df.empty:
        return auction_df

    normal = add_exchange_times(normal, exchange_open, exchange_close)
    normal = filter_time(normal, exchange_open, exchange_close)
    resampled = aggregate_trades(normal, interval)

    # Removing the day from the index and converting the time to a string
    resampled.index = resampled.index.strftime('%H:%M:%S')

    # This sets the opening auction at the beginning and closing auction at the end of the resampled data
    resampled = pd.concat([
        auction_df.iloc[[0]],
        resampled,
        auction_df.iloc[[1]]
    ])

    return fix_missing(resampled)

def process_file(file: str, save_dir: str, exchange_open: pd.Timestamp, exchange_close: pd.Timestamp,
                 interval: str) -> None:
    """

    Parameters
    ----------
    file : Parquet file with trade data at some day of a stock.
    save_dir : Directory to save the aggregated trade data files.
    exchange_open : Opening time of the exchange in the format 'HH:MM:SS'
    exchange_close : Closing time of the exchange in the format 'HH:MM:SS'
    interval : Aggregation interval using a format that is supported by Pandas resample function

    Returns
    -------
    None
    """

    df = pd.read_parquet(file)
    df.columns = ['time', 'price', 'quantity', 'flag']
    day = file.split('/')[-1].split('.')[0]

    df = process_df(df, exchange_open, exchange_close, interval)

    if not df.empty:
        df.to_parquet(f'{save_dir}/{day}.parquet')

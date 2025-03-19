import pandas as pd
import gzip
import os
import multiprocessing
from itertools import islice
from tqdm import tqdm

class TradeSplitter:
    """
    Class to load all zipped trade data files from a given directory. This class splits one trade file into multiple
    files based on the day. The data is then saved in a new directory.
    """
    def __init__(self, trades_dir: str, save_dir: str, stock_info_path: str, n_cores: int = None, core_files: int = 1):
        """
        Initialize the TradeSplitter object.

        Parameters
        ----------
        trades_dir : Directory containing the zipped trade data files.
        save_dir : Directory to save the unzipped trade data files.
        stock_info_path : Path to the file containing the stock information.
        n_cores : Number of cores to use for multiprocessing. Default is None, which uses all available cores
        core_files : Number of files to process per core per batch. Default is 1
        """
        self.trades_dir = trades_dir
        self.save_dir = save_dir

        self.stock_info_df = pd.read_csv(stock_info_path)

        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
        self.core_files = core_files
        self.batch_size = self.n_cores * self.core_files

        if not os.path.exists(self.save_dir):
            print(f'Creating directory {self.save_dir}')
            os.makedirs(self.save_dir)

    def split_batch(self, files: list[str]) -> None:
        """
        Splits a batch of zipped trade data files by day and saves them in a new directory.

        Parameters
        ----------
        files : Files to process in this batch

        Returns
        -------
        None
        """

        # Get dictionary with all dataframes
        items = [(file, self.save_dir, self.stock_info_df) for file in files]
        with multiprocessing.Pool(self.n_cores) as pool:
            dict_list = pool.starmap(process_file, items)
        combined_dict = merge_dictionaries(dict_list)

        # Each core saves its assigned DataFrames
        dict_splits = split_dict(combined_dict, self.n_cores)
        with multiprocessing.Pool(self.n_cores) as pool:
            pool.map(save_files, dict_splits)

    def split_all(self) -> None:
        """
        Splits all the zipped trade data files by day and saves them in a new directory.

        Returns
        -------
        None
        """

        all_files = [os.path.join(self.trades_dir, f) for f in os.listdir(self.trades_dir) if f.endswith(".gz")]

        file_batches = [all_files[i:i + self.batch_size] for i in range(0, len(all_files), self.batch_size)]

        for batch in tqdm(file_batches):
            self.split_batch(batch)

def load_from_gzip(zip_file: str) -> pd.DataFrame:
    """
    Load a zipped trade data file.

    Parameters
    ----------
    zip_file : Path to the zipped trade data file.

    Returns
    -------
    Dataframe containing the trade data.
    """
    with gzip.open(zip_file, 'rt') as f:
        df = pd.read_csv(f, header=0, names=['time', 'price', 'quantity', 'flag'], engine='pyarrow')
    return df

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

def combine_rows(df: pd.DataFrame) -> pd.DataFrame:
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
    ).reset_index()

def has_auctions(df: pd.DataFrame) -> bool:
    """
    This function checks if the given dataframe has a opening and closing auction.

    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    Boolean indicating if the dataframe has opening and closing auction.
    """
    auction_df = df[df['flag'] == 'AUCTION']
    n_auction_prices = auction_df['price'].nunique()
    return n_auction_prices == 2

def group_by_day(df: pd.DataFrame, save_dir) -> dict[str, pd.DataFrame]:
    """
    Group the trade data by day. The keys will be the save path for the parquet file.

    Parameters
    ----------
    df : Dataframe containing the trade data.
    save_dir : Directory to save the parquet files.

    Returns
    -------
    Dictionary containing the trade data grouped by day.
    """
    groups = df.groupby(df['t'].dt.date)

    grouped_dict = {}
    for day, group in groups:
        if not has_auctions(group):
            continue

        key = os.path.join(save_dir, f'{day.strftime("%Y-%m-%d")}.parquet')
        grouped_dict[key] = group

    return grouped_dict

def get_id(file_name: str) -> int:
    """
    Get the id corresponding to the stock from the file name.

    Parameters
    ----------
    file_name : Name of the file.

    Returns
    -------
    ID of the stock
    """
    # ID is after the last underscore
    stock_id = file_name.split('_')[-1]

    # removes file extension
    return int(
        stock_id.split('.')[0]
    )

def get_exchange_name(stock_bloomberg: str) -> str:
    """
    Get the exchange name from the stock bloomberg name.

    Parameters
    ----------
    stock_bloomberg : Name of the stock.

    Returns
    -------
    Name of the exchange.
    """
    exchange = stock_bloomberg.split(' ')[1]
    return exchange

def stock_info(stock_id: int, exchange_df: pd.DataFrame) -> tuple[str, str]:
    """
    Get the exchange and stock name from the id.

    Parameters
    ----------
    stock_id : ID of the stock.
    exchange_df : Dataframe containing the exchange data.

    Returns
    -------
    Tuple containing the exchange and stock name.
    """
    # Finding the row that corresponds to the stock_id
    stock_row = exchange_df[exchange_df['syd'] == stock_id]

    stock_bloomberg = stock_row['bb'].values[0]
    exchange_name = get_exchange_name(stock_bloomberg)
    stock_name = stock_row['name'].values[0]

    return exchange_name, stock_name

def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function applies all the transformations to the trade data. The transformations are:
    - Load the data
    - Convert the time column to a datetime object
    - Remove rows which are not normal or auction trades
    - Combine rows which occurred at the same time and have the same trading price
    Parameters
    ----------
    df : Dataframe containing the trade data.

    Returns
    -------
    Dataframe with the transformations applied.
    """
    df = convert_to_time(df)
    df = filter_rows(df)
    df = combine_rows(df)
    return df

def process_file(file: str, save_dir: str, exchange_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Process a single zipped trade data file. This function starts by unzipping the file, then loads the data,
    converts the time column to a datetime object, removes rows which are not normal or auction trades, groups the data
    by day, and then returns a dictionary where the keys are the save paths for the parquet files.

    Parameters
    ----------
    file : Name of the zipped trade data file.
    save_dir : Directory to save the unzipped trade data files.
    exchange_df : Dataframe containing the exchange data.

    Returns
    -------
    Dictionary containing the trade data grouped by day.
    """

    df = load_from_gzip(file)
    df = apply_transformations(df)

    stock_id = get_id(file)
    exchange_name, stock_name = stock_info(stock_id, exchange_df)

    # Exchange directory
    save_dir = os.path.join(save_dir, exchange_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a directory for the stock if it does not exist
    stock_name = stock_name.replace(' ', '_')
    save_dir = os.path.join(save_dir, stock_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return group_by_day(df, save_dir)

def merge_dictionaries(dict_list: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """
    Merge multiple dictionaries into one.

    Parameters
    ----------
    dict_list : List of dictionaries to merge.

    Returns
    -------
    Merged dictionary.
    """
    merged_dict = {}
    for d in dict_list:
        merged_dict.update(d)
    return merged_dict

def split_dict(d: dict, n: int) -> list[dict]:
    """
    Splits a dictionary into n roughly equal parts.

    Parameters
    ----------
    d : Dictionary to split.
    n : Number of parts to split the dictionary into.

    Returns
    -------
    List of dictionaries.
    """
    it = iter(d.items())
    return [{k: v for k, v in islice(it, len(d) // n + (i < len(d) % n))} for i in range(n)]

def save_files(file_dict: dict[str, pd.DataFrame]) -> None:
    """
    Save the trade data to parquet files.

    Parameters
    ----------
    file_dict : Dictionary containing the trade data.

    Returns
    -------
    None
    """
    for save_path, df in file_dict.items():
        df.to_parquet(save_path, index=False)
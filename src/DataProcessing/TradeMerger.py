import pandas as pd
import multiprocessing
import os
import numpy as np
import fastparquet
import datetime
from tqdm import tqdm

class TradeMerger:
    """
    This class merges all trade files from a given year and month into a singular file across all stocks. This allows
    for walk-forward testing and training of the model.
    """
    def __init__(self, trades_dir: str, save_dir: str, sequence_size: int = None, n_cores: int = None) -> None:
        """
        Initialize the TradeMerger class

        Parameters
        ----------
        trades_dir : Directory where all the trades are stored
        save_dir : Directory where all the results will be saved
        sequence_size : Number of minutes in the sequence to consider. Does not have to be provided,
        however, it will use the smallest sequence size from a given dataset
        """
        self.trades_dir = trades_dir
        self.save_dir = save_dir
        self.sequence_size = sequence_size
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
        if sequence_size is None:
            self.find_sequence_size()


    def find_sequence_size(self) -> None:
        """
        Find the smallest sequence size from the given dataset. This is used to determine the sequence size for the
        entire dataset by finding the minimum sequence size across all stocks. It is assumed that the sequence size
        for a stock is constant across all files.

        Returns
        -------
        None
        """
        stocks = os.listdir(self.trades_dir)
        sequence_sizes = []
        for stock in stocks:
            stock_path = os.path.join(self.trades_dir, stock)

            # Only need first file
            files = os.listdir(stock_path)
            if len(files) == 0:
                continue

            first_file = os.listdir(stock_path)[0]
            file_path = os.path.join(stock_path, first_file)

            # Subtract 2 to account for the auctions
            sequence_size = get_parquet_row_count(file_path) - 2
            sequence_sizes.append(sequence_size)

        self.sequence_size = min(sequence_sizes)

    def process_all(self) -> None:
        """
        Process all trades for each stock within the given directory. This will merge all trades for each stock into a
        single file.

        Returns
        -------
        None
        """
        # Get all stock
        stocks = os.listdir(self.trades_dir)
        stocks.sort()

        # Add the path to the stock trades
        stocks = [os.path.join(self.trades_dir, stock) for stock in stocks]

        year_ranges = np.arange(2021, 2023)
        month_ranges = np.arange(1, 13)
        with tqdm(total=len(year_ranges) * len(month_ranges)) as pbar:
            for year in range(2021, 2023):
                for month in range(1, 13):
                    # Getting all the files that we need to process
                    files = []
                    for stock in stocks:
                        stock_files = get_files(stock, year, month)
                        if len(stock_files) > 0:
                            files.append(stock_files)

                    # If there are no files to process, skip
                    if len(files) == 0:
                        pbar.update(1)
                        continue


                    # Processing and saving
                    save_dir = os.path.join(self.save_dir, f'{year}-{month}')
                    process_month(files, save_dir, self.sequence_size, self.n_cores)

                    pbar.update(1)

def process_month(files: list[list[str]], save_dir, sequence_length: int, n_cores: int) -> None:
    """
    Process all trades for a given month and year. This will merge all trades for each stock into a single file.

    Parameters
    ----------
    files : List which contains the files to process for each stock as its own list
    save_dir : Directory where all the results will be saved
    sequence_length : Number of minutes in the sequence
    n_cores : Number of cores to use for processing

    Returns
    -------
    None
    """

    items = [(stock, sequence_length) for stock in files]

    # Create a pool of workers
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.starmap(process_stock_files, items)

    # Merge all the results into a single list
    merged_results = merge_lists(results)

    # Combine all results to singular dictionary for each variable
    combined_results = combine_results(merged_results)

    save_results(combined_results, save_dir)

def combine_results(results: list[dict[str, np.ndarray]]) -> dict[str, pd.DataFrame]:
    """
    Combine all the results into a single dictionary for each variable. This is used to save the results to a single
    file.

    Parameters
    ----------
    results : List of dictionaries with the processed data

    Returns
    -------
    List of dictionaries with the processed data for each variable
    """

    combined_results = {}

    # Combine all results into a single dictionary for each variable
    for result in results:
        for key, value in result.items():
            if key not in combined_results:
                combined_results[key] = []

            combined_results[key].append(value)

    # Working with dataframes allow for easier manipulation and saving
    for key, value in combined_results.items():
        combined_results[key] = pd.DataFrame(value)

    return combined_results

def save_results(results: dict[str, pd.DataFrame], save_dir: str) -> None:
    """
    Save each variable of results to a separate file.

    Parameters
    ----------
    results : List of dictionaries with the processed data
    save_dir : Directory where the results will be saved

    Returns
    -------
    None
    """

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for key, value in results.items():
        # Save the dataframe to a parquet file
        file_path = os.path.join(save_dir, f'{key}.parquet')
        value.to_parquet(file_path, index=False)

def get_date(file: str) -> datetime.date:
    """
    Get the date from the file name

    Parameters
    ----------
    file : File name to get the date from

    Returns
    -------
    Date of the file
    """
    # Remove the extension
    file = file.split('.')[0]

    return pd.Timestamp(file).date()

def get_files(stock_path: str, year: int, month: int) -> list[str]:
    """
    Get all the files for a given stock in a given year and month. It will also include one file before the given month
    which may be needed to get the previous day trades.

    Parameters
    ----------
    stock_path : Path to the stock trades
    year : Year to get the files from
    month : Month to get the files from

    Returns
    -------
    List with the file names
    """
    files = os.listdir(stock_path)
    files.sort()

    # Find the first and last indices which are in the given month
    first_index = None
    last_index = None
    for i, file in enumerate(files):
        date = get_date(file)
        if date.year == year and date.month == month:
            if first_index is None:
                first_index = i
            last_index = i

    if first_index is None or last_index is None:
        return []

    # Adding the stock path to the files
    first_index = max(0, first_index - 1)
    files = files[first_index:last_index + 1]
    files = [os.path.join(stock_path, file) for file in files]

    return files

def compute_current_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the current return relative to the opening price

    Parameters
    ----------
    df : DataFrame with the trades

    Returns
    -------
    DataFrame with the current returns
    """
    prices = df['price'].values
    current_prices = prices[:-1]
    opening_price = prices[0]

    return np.log(current_prices / opening_price)

def compute_overnight_return(df: pd.DataFrame, previous_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the overnight return by comparing the closing price of the previous day with the opening price of the
    current day.

    Parameters
    ----------
    df : DataFrame with the trades for the current day
    previous_df : DataFrame with the trades for the previous day

    Returns
    -------
    DataFrame with the overnight returns
    """
    opening_price = df['price'].iloc[0]
    closing_price = previous_df['price'].iloc[-1]

    return np.log(opening_price / closing_price)
    
def compute_previous_returns(previous_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the returns of the previous day from closing price to now

    Parameters
    ----------
    previous_df : DataFrame with the trades for the previous day

    Returns
    -------
    DataFrame with the previous returns
    """
    prices = previous_df['price'].values
    current_prices = prices[:-1]
    closing_price = prices[-1]
    returns = np.log(closing_price / current_prices)

    return returns

def compute_relative_traded_volume(df: pd.DataFrame, previous_df: pd.DataFrame) -> np.ndarray:
    """
    Compute how the current traded volume compares to the previous day closing traded volume

    Parameters
    ----------
    df : DataFrame with the trades for the current day
    previous_df : DataFrame with the trades for the previous day

    Returns
    -------
    DataFrame with the relative traded volume
    """
    current_volumes = df['quantity'].values
    previous_closing_volume = previous_df['quantity'].iloc[-1]
    current_traded_volume = current_volumes[:-1].cumsum()

    return current_traded_volume / previous_closing_volume

def compute_closing_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the closing returns for the current day

    Parameters
    ----------
    df : DataFrame with the trades for the current day

    Returns
    -------
    DataFrame with the closing returns
    """
    prices = df['price'].values
    closing_price = prices[-1]
    current_prices = prices[:-1]

    return np.log(closing_price / current_prices)

def set_sequence_length(result: dict[str, np.ndarray], sequence_length: int) -> dict[str, np.ndarray]:
    """
    Set the sequence length for the data

    Parameters
    ----------
    result : Dictionary with the processed data
    sequence_length : Number of minutes in the sequence

    Returns
    -------
    Dict of the processed data with the correct sequence length
    """

    result['previous_returns'] = result['previous_returns'][-sequence_length:]
    result['current_returns'] = result['current_returns'][-sequence_length:]
    result['relative_traded_volume'] = result['relative_traded_volume'][-sequence_length:]
    result['closing_returns'] = result['closing_returns'][-sequence_length:]


    return result

def process_file(df: pd.DataFrame, previous_df: pd.DataFrame, sequence_length: int) -> dict[str, np.ndarray]:
    """
    Process a file to get the data in the correct format.

    Parameters
    ----------
    df : DataFrame to process
    previous_df : DataFrame with the previous day trades
    sequence_length : Number of minutes in the sequence

    Returns
    -------
    Dictionary with the processed data for each column
    """

    overnight_return = compute_overnight_return(df, previous_df)
    previous_returns = compute_previous_returns(previous_df)
    current_returns = compute_current_returns(df)
    relative_traded_volume = compute_relative_traded_volume(df, previous_df)
    closing_returns = compute_closing_returns(df)

    result = {
        'overnight_returns': overnight_return,
        'previous_returns': previous_returns,
        'current_returns': current_returns,
        'closing_returns': closing_returns,
        'relative_traded_volume': relative_traded_volume
    }

    return set_sequence_length(result, sequence_length)


def is_previous_day(previous_date, date):
    """
    Check if previous_date is the previous trading day to date

    Parameters
    ----------
    previous_date : Date to check if it is the previous trading day
    date : Date to check if it is the current trading day

    Returns
    -------
    Boolean indicating if previous_date is the previous trading day to date
    """
    # Convert both to timestamps
    previous_date = pd.Timestamp(previous_date).date()
    date = pd.Timestamp(date).date()

    # On mondays we need to subtract 3 days to get the previous trading day
    days_to_subtract = 3 if date.weekday() == 0 else 1
    previous_trading_day = date - pd.Timedelta(days=days_to_subtract)

    return previous_date == previous_trading_day


def process_stock_files(files: list[str], sequence_length: int) -> list[dict[str, np.ndarray]]:
    """
    Process all trades of a given stock

    Parameters
    ----------
    files : List of files to process
    sequence_length : Number of minutes in the sequence

    Returns
    -------
    List with the processed data for each day
    """
    # We need to keep track of the previous date to check if we are in the same trading day
    first_file = files[0]
    previous_df = pd.read_parquet(first_file)

    # We need to remove path and extension from the file name to get the date
    first_file = first_file.split('/')[-1]
    previous_date = first_file.split('.')[0]

    dataset = []
    for file in files[1:]:
        df = pd.read_parquet(file)
        date = file.split('/')[-1]
        date = date.split('.')[0]

        # We need yesterday's data to compute the overnight return
        if is_previous_day(previous_date, date):
            result = process_file(df, previous_df, sequence_length)
            dataset.append(result)

        previous_date = date
        previous_df = df

    return dataset

def merge_lists(results: list) -> list:
    """
    Merge all the lists into a single list. This is used to combine the results from all the stocks into a single

    Parameters
    ----------
    results : List of lists to merge

    Returns
    -------
    List with all the elements from the input lists
    """
    merged = []
    for result in results:
        if result is not None:
            merged.extend(result)

    return merged

def get_parquet_row_count(file_path: str) -> int:
    """
    Get the number of rows in a Parquet file without loading the entire file.

    Parameters
    ----------
    file_path : Path to the Parquet file.

    Returns
    -------
    int
        Number of rows in the Parquet file.
    """
    parquet_file = fastparquet.ParquetFile(file_path)

    return parquet_file.info['rows']

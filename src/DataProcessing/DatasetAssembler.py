import pandas as pd
import multiprocessing
import os
import numpy as np
import fastparquet
import datetime
from tqdm import tqdm

date_type = list[int]

class DatasetAssembler:
    """
    This class merges all aggregated trade files from a given year and month into a singular file across all stocks.
    This allows for walk-forward testing and training of the model per month.
    """
    def __init__(self, trades_dir: str, save_dir: str, sequence_size: int = None, n_cores: int = None) -> None:
        """
        Initialize the DatasetAssembler class with the given parameters.

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

        # We lose one value from opening which is already in the overnight return
        self.sequence_size = min(sequence_sizes) - 1
        print(f'Using sequence size of {self.sequence_size} minutes')

    def process_all(self, start_date: date_type, end_date: date_type) -> None:
        """
        Process all trades for each stock within the given directory. This will merge all trades for each stock into a
        single file.

        Parameters
        ----------
        start_date : Start date to process the trades from. Given as [year, month]
        end_date : End date to process the trades to. Given as [year, month]

        Returns
        -------
        None
        """
        # Get all stocks
        stocks = os.listdir(self.trades_dir)
        stocks.sort()

        # Add the path to the stock trades
        stocks = [os.path.join(self.trades_dir, stock) for stock in stocks]

        dates = generate_dates(start_date, end_date)
        with tqdm(total=len(dates)) as pbar:
            for year, month in dates:
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

def generate_dates(start_date: date_type, end_date: date_type) -> list[date_type]:
    """
    Generate a list of dates between the start and end date. The dates are given as [year, month].

    Parameters
    ----------
    start_date : Start date to process the trades from. Given as [year, month]
    end_date : End date to process the trades to. Given as [year, month]

    Returns
    -------
    List of dates between the start and end date
    """
    start_year, start_month = start_date
    end_year, end_month = end_date

    dates = []
    year, month = start_year, start_month

    # Checks if we are before the end year, or before the end month in the same year
    while (year < end_year) or (year == end_year and month <= end_month):
        dates.append([year, month])
        month += 1
        if month > 12:
            month = 1
            year += 1

    return dates

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

def process_month(files: list[list[str]], save_dir, sequence_length: int, n_cores: int) -> None:
    """
    Process all trades for a given month and year. This will merge all trades for each stock into a single file.

    Parameters
    ----------
    files : List which contains the files to process for each stock as its own list
    save_dir : Directory where all the results will be saved
    sequence_length : Number of minutes in the sequence
    n_cores : Number of cores to use for processing
    """

    items = [(stock, sequence_length) for stock in files]

    # Create a pool of workers and have them process each stock in parallel
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.starmap(process_stock_files, items)

    # Merge all the results into a single list
    merged_results = merge_lists(results)

    # Combine all results to singular dictionary for each variable
    combined_results = combine_results(merged_results)

    save_results(combined_results, save_dir)

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

    # We need the previous day trades to compute the overnight return
    previous_date = get_date(first_file)

    dataset = []

    # Since we already have first file, we start from the second file
    for file in files[1:]:
        df = pd.read_parquet(file)

        # The dataset may contain NaN values, which breaks the processing
        if has_nan(df):
            continue

        date = get_date(file)
        if process_current_file(df, previous_date, date, sequence_length):
            result = process_file(df, previous_df, sequence_length)
            dataset.append(result)

        # Update for next iteration
        previous_date = date
        previous_df = df

    return dataset

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
    # Removing the path
    file_name = file.split('/')[-1]
    # Remove the extension
    date = file_name.split('.')[0]

    # Get the date from the file name
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()

    return date

def has_nan(df: pd.DataFrame):
    """
    Check if the dataframe has any NaN values

    Parameters
    ----------
    df : DataFrame to check

    Returns
    -------
    Boolean indicating if the dataframe has any NaN values
    """
    # First and last are not used
    df = df.iloc[1:-1].copy()
    return df.isnull().values.any()

def process_current_file(df: pd.DataFrame, previous_date: datetime.date, date: datetime.date, sequence_length: int) -> bool:
    """
    Check if we should process the current file. This is done by checking if the previous date is a trading day before
    the current date, and if the length of this dataframe is greater or equal to the sequence length.

    Parameters
    ----------
    df : DataFrame to check
    previous_date : Date to check if it is the previous trading day
    date : Date to check if it is the current trading day
    sequence_length : Number of minutes in the sequence

    Returns
    -------
    Boolean representing if the current file should be processed
    """
    if not is_previous_day(previous_date, date):
        return False
    if len(df) < sequence_length + 2:
        return False

    return True

def is_previous_day(previous_date: datetime.date, date: datetime.date):
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
    # On mondays we need to subtract 3 days to get the previous trading day
    days_to_subtract = 3 if date.weekday() == 0 else 1
    previous_trading_day = date - pd.Timedelta(days=days_to_subtract)

    return previous_date == previous_trading_day

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
    """

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for key, value in results.items():
        # Check for nans before saving
        if value.isnull().values.any():
            raise ValueError(f"DataFrame {key} contains NaN values")

        # Save the dataframe to a parquet file
        file_path = os.path.join(save_dir, f'{key}.parquet')
        value.to_parquet(file_path, index=False)


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
    # Dictionary that is used to store each variable
    result = {}

    # Time independent variables
    overnight_return = compute_overnight_return(df, previous_df)
    current_return = compute_returns(df)
    previous_return = compute_returns(previous_df)

    result['overnight_return'] = np.array([overnight_return])
    result['current_return'] = np.array([current_return])
    result['previous_return'] = np.array([previous_return])

    # Since the length of the df can be greater than the sequence length, we need to combine the first few rows
    df = match_sequence_length(df, sequence_length)

    # Check if nan in df
    if df.isnull().values.any():
        print("DataFrame contains NaN values")
        raise ValueError("DataFrame contains NaN values")

    # Normalize the prices
    prices_df = normalize_prices(df)
    result.update({feature: prices_df[feature].values for feature in prices_df.columns})

    # Normalize the limit order book volume
    lob_volume_df = normalize_lob_volume(df)
    result.update({feature: lob_volume_df[feature].values for feature in lob_volume_df.columns})

    # Normalize the volume
    volume_df = normalize_volume(df)
    result['quantity'] = volume_df.values

    # Check if any nan values are present
    for feature in result:
        if np.isnan(result[feature]).any():
            print(f"DataFrame {feature} contains NaN values after processing")
            raise ValueError(f"DataFrame {feature} contains NaN values")

    return result

def compute_overnight_return(df: pd.DataFrame, previous_df: pd.DataFrame) -> float:
    """
    Compute the overnight return for the given dataframe. This return is defined as the opening price of today
    relative to the closing price of the previous day.

    Parameters
    ----------
    df : DataFrame to compute the overnight return
    previous_df : DataFrame with the previous day trades

    Returns
    -------
    Overnight return
    """
    # Opening price is first index, and closing price is last index
    today_opening_price = df.iloc[0]['vwap']
    previous_closing_price = previous_df.iloc[-1]['vwap']

    overnight_return = (today_opening_price - previous_closing_price) / previous_closing_price

    return overnight_return

def compute_returns(df: pd.DataFrame) -> float:
    """
    Compute the returns for the given dataframe. This return is defined as the closing price of today
    relative to the current vwap price. This is the prediction target.

    Parameters
    ----------
    df : DataFrame to compute the returns

    Returns
    -------
    Current return
    """
    current_price = df.iloc[-2]['vwap']
    closing_price = df.iloc[-1]['vwap']

    current_return = (closing_price - current_price) / current_price

    return current_return

def match_sequence_length(df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    """
    Match the sequence length of the dataframe to the given sequence length. This is done by combining the first
    few rows of the dataframe to match the sequence length. This is done to avoid losing data when the
    dataframe is greater than the sequence length.

    Parameters
    ----------
    df : DataFrame to match the sequence length
    sequence_length : Number of minutes in the sequence

    Returns
    -------
    Dataframe with the correct sequence length
    """
    # We only need the trade-related rows
    df = df.iloc[1:-1].copy()

    # Additional check
    if df.shape[0] < sequence_length:
        raise ValueError(f"DataFrame has less than {sequence_length} rows which has {len(df)} rows")

    # No point in combining if the sequence length already matches
    if df.shape[1] == sequence_length:
        return df

    # Combine the first few rows to match the sequence length
    rows_to_combine = df.shape[0] - sequence_length + 1
    combined_rows = combine_rows(df.iloc[:rows_to_combine])

    # Remove all the rows
    df = df.iloc[rows_to_combine:]

    # Add the combined row to the beginning of the dataframe
    df = pd.concat([combined_rows, df], ignore_index=True)

    return df

def combine_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the given rows to one row.

    Parameters
    ----------
    rows : DataFrame to combine

    Returns
    -------
    Combined DataFrame
    """
    # Take copy of the last row
    combined_row = rows.iloc[-1:].copy()

    combined_row['open'] = rows.iloc[0]['open']
    combined_row['high'] = rows['high'].max()
    combined_row['low'] = rows['low'].min()
    combined_row['quantity'] = rows['quantity'].sum()

    # VWAP price can only be computed if there is a quantity
    if (combined_row['quantity'] > 0).values.any():
        volume_weighted_price = rows['vwap'] * rows['quantity']
        combined_row['vwap'] = volume_weighted_price.sum() / combined_row['quantity']

    return combined_row

def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the prices in the given dataframe. This is a min-max normalization based on the VWAP prices.

    Parameters
    ----------
    df : DataFrame to normalize

    Returns
    -------
    Normalized prices
    """
    vwap_prices = df['vwap']
    vwap_min = vwap_prices.min()
    vwap_max = vwap_prices.max()

    # Columns to be normalized
    trade_columns = ['close', 'open', 'high', 'low', 'vwap']
    lob_columns = [f'bb{i}' for i in range(1, 6)] + [f'ba{i}' for i in range(1, 6)]
    columns = trade_columns + lob_columns
    df = df[columns].copy()

    # Normalize the columns
    for column in columns:
        df[column] = (df[column] - vwap_min) / (vwap_max - vwap_min)

    return df

def normalize_lob_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the limit order book volume in the given dataframe. This normalization combines the total volume of the
    bid and ask side and divides each volume by the total volume. This keeps relative volume information while
    removing the absolute volume information.

    Parameters
    ----------
    df : DataFrame to normalize

    Returns
    -------
    Normalized limit order book volume
    """
    # Columns
    bid_columns = [f'bbvol{i}' for i in range(1, 6)]
    ask_columns = [f'bavol{i}' for i in range(1, 6)]
    columns = bid_columns + ask_columns

    # Only keep the columns we need
    df = df[columns].copy()

    # Normalization is based on the total volume of the bid and ask side
    total_volume = df.sum(axis=1)

    # Replace total volume with 1 to avoid division by zero
    total_volume[total_volume == 0] = 1

    # Normalize the columns
    for column in columns:
        df[column] = df[column] / total_volume

    return df

def normalize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the volume in the given dataframe. This is done by computing the total traded volume and
    dividing each volume by the total volume. This keeps relative volume information while removing the absolute volume
    information.

    Parameters
    ----------
    df : DataFrame to normalize

    Returns
    -------
    Normalized trade volume
    """

    column = 'quantity'
    df = df[column].copy()

    # Total volume
    total_volume = df.sum()

    # Compute relative volume
    df = df / total_volume

    return df

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

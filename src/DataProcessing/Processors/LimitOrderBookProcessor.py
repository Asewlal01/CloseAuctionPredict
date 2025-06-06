from DataProcessing.Processors import BaseProcessor
import pandas as pd
import numpy as np
import datetime
import multiprocessing
from typing import Hashable


class LimitOrderBookProcessor(BaseProcessor):
    """
    This class processes limit order book data files. It inherits from the FileProcessor class and implements the
    process_file method.
    """
    def __init__(self, file_path: str, max_price: int):
        """
        Initialize the LimitOrderBookProcessor with the path to the file to be processed.

        Parameters
        ----------
        file_path : Path to the file to be processed.
        max_price : Maximum price allowed in the dataframe. If any price exceeds this value, it is considered invalid.
        """

        super().__init__(file_path)
        self.max_price = max_price

    def process_file(self):
        """
        Process the trade data file. This method is called during initialization.
        """
        self.aggregated_data = process_all_days(self.df, self.max_price)

def process_all_days(df: pd.DataFrame, max_price) -> dict[Hashable | datetime.date, pd.DataFrame]:
    """
    Process all the days of a given dataframe. Each day is processed in parallel using multiprocessing and the
    process_day function. The results are aggregated into a dictionary with dates as keys and DataFrames as values.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    max_price : Maximum price allowed in the dataframe. If any price exceeds this value, it is considered invalid.

    Returns
    -------
    Dictionary with dates as keys and DataFrames as values. Each DataFrame contains the aggregated trade data for that date.
    """
    # Nothing to process
    if df.empty:
        return {}

    # The nature column is not needed for the processing
    df = df.drop(columns=['nature'])

    # Group into days
    groups = df.groupby(df['t'].dt.date)

    # Use multiprocessing to process each group in parallel
    items = [(group, date, max_price) for date, group in groups]
    with multiprocessing.Pool() as pool:
        processed_groups = pool.starmap(process_day, items)

    # Concatenate the results into a dictionary
    processed_groups = {date: data for date, data in processed_groups if not data.empty}

    return processed_groups

def process_day(df: pd.DataFrame, date: datetime.date, max_price) -> tuple[datetime.date, pd.DataFrame]:
    """
    Process a single day's data. The function checks for missing values, fixes them, and sets the time column as the
    index. If the data is invalid, it returns an empty DataFrame.

    Parameters
    ----------
    df : Dataframe with the data to be processed.
    date: Date of the data to be processed.
    max_price: Maximum price allowed in the dataframe. If any price exceeds this value, it is considered invalid.

    Returns
    -------
    Dataframe with the aggregated data for the day.
    """

    # Nothing to process
    if df.empty:
        return date, pd.DataFrame()

    # Check if we have any missing values
    if is_invalid(df):
        return date, pd.DataFrame()

    # Fix all the missing data
    df = fix_missing(df)

    # Check if we have any missing values
    if has_missing(df, max_price):
        return date, pd.DataFrame()

    # Set the time column as the index
    df = df.set_index('t')

    return date, df

def is_invalid(df: pd.DataFrame) -> bool:
    """
    Check if there are any missing values in the dataframe. It is expected that every observation has at least the first
    level on the bid and ask side. If there are any missing values for the first level, the dataframe is considered to be
    invalid.

    Parameters
    ----------
    df : DataFrame to check.

    Returns
    -------
    Boolean indicating whether the dataframe is invalid or not.
    """

    # The columns we are interested in
    columns = ['bb1', 'bbvol1', 'ba1', 'bavol1']
    df_columns = df[columns]

    # Checks for missing or invalid prices
    invalid = df_columns.isna() * df_columns.eq(0)

    # Checks if there are any missing values per column
    invalid = invalid.any()

    # Across all columns
    return invalid.any()

def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing data in the dataframe. It is assumed that the first level is always present, as their prices
    are used to fill the missing values in the other levels the missing values. The quantities are set to zero if they
    are missing.

    Parameters
    ----------
    df : DataFrame to fix.

    Returns
    -------
    DataFrame with fixed data.
    """

    # Fixing rows that are partially empty. Assumed to have 5 LOB levels
    levels = 5
    for i in range(2, levels + 1):
        for side in ['b', 'a']:
            price_col = f'b{side}{i}'
            quantity_col = f'b{side}vol{i}'
            prev_price_col = f'b{side}{i-1}'

            # May be zero or nan, hence we put everything to nan
            df[price_col] = df[price_col].replace(0, np.nan)

            # Price is same as previous level
            df[price_col] = df[price_col].fillna(df[prev_price_col])

            # Quantity is zero
            df[quantity_col] = df[quantity_col].fillna(0)

    return df

def has_missing(df: pd.DataFrame, max_price: int) -> bool:
    """
    Check if the dataframe has any missing values. This function is used as a final check after processing the data.
    If it returns True, there is likely something wrong with the dataset, which cannot be fixed by the previous methods.

    Parameters
    ----------
    df : DataFrame to check.
    max_price : Maximum price allowed in the dataframe. If any price exceeds this value, it is considered invalid.

    Returns
    -------
    bool : True if the dataframe has missing values, False otherwise.
    """

    # Price checking
    for i in range(1, 6):
        for side in ['b', 'a']:
            price_col = f'b{side}{i}'

            # Nan values means missing values
            if df[price_col].isna().any():
                return True

            # Values at and below zero imply missing values
            if df[price_col].le(0).any():
                return True

            # The price should not exceed the maximum price
            if df[price_col].gt(max_price).any():
                return True

    return False
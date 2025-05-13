from DataProcessing.Processors import BaseProcessor
import pandas as pd
import datetime
import multiprocessing

class AuctionProcessor(BaseProcessor):
    """
    This class handles the trade files to obtain the auction data.
    """

    def process_file(self):
        """
        Process the trade data file. This method is called during initialization.
        """
        self.aggregated_data = process_all_days(self.df)


def process_all_days(df: pd.DataFrame) -> dict[datetime.date, pd.DataFrame]:
    """
    Process all days in the dataframe to get the auction data.

    Parameters
    ----------
    df : DataFrame containing the trade data.

    Returns
    -------
    DataFrame containing the aggregated trade data.
    """
    if df.empty:
        return {}

    # Group the dataframe by date
    grouped_by_date = df.groupby(df['t'].dt.date)

    items = [(group_df, date) for date, group_df in grouped_by_date]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_day, items)

    # Filter out empty results
    results = {date: data for date, data in results if not data.empty}

    return results


def process_day(df: pd.DataFrame, date: datetime.date) -> tuple[datetime.date, pd.DataFrame]:
    """
    Process a single day of data to get the auction data.

    Parameters
    ----------
    df : DataFrame containing the trade data for a single day.
    date : Date of the data to be processed.

    Returns
    -------
    Auction DataFrame.
    """
    # Filter the dataframe to only include rows with the 'flag' column set to 'NORMAL' or 'AUCTION'
    df = filter_flags(df)

    # Get the indices of the first and last intraday data
    indices = get_indices(df)

    # Check if the dataframe has auction data
    if not has_auctions(df, indices):
        return date, pd.DataFrame()

    # Get the auction data
    auction_data = get_auction_data(df, indices)

    return date, auction_data

def filter_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe to only include rows with the 'flag' column set to 'NORMAL' or 'AUCTION'.

    Parameters
    ----------
    df : DataFrame to filter.

    Returns
    -------
    DataFrame : Filtered DataFrame.
    """
    return df[df['flag'].isin(['NORMAL', 'AUCTION'])].reset_index(drop=True)

def get_indices(df: pd.DataFrame) -> tuple[int, int]:
    """
    Get the indices of the first and last intraday data in the dataframe.
    This is used to determine the opening and closing auction data.
    Parameters
    ----------
    df: DataFrame to get the indices from.

    Returns
    -------
    Indices of first and last intraday data.
    """
    # Filter the dataframe to only include rows with the 'flag' column set to 'NORMAL'
    intraday_df = df[df['flag'] == 'NORMAL']

    # Check if the dataframe is empty
    if intraday_df.empty:
        return 0, 0

    # Get the first and last index of the intraday data
    first_idx = intraday_df.index[0]
    last_idx = intraday_df.index[-1]

    return first_idx, last_idx

def has_auctions(df: pd.DataFrame, indices: tuple[int, int]) -> bool:
    """
    Check if the dataframe has auction data.

    Parameters
    ----------
    df : DataFrame to check.
    indices : Indices of the first and last intraday data.

    Returns
    -------
    bool : True if the dataframe has auction data, False otherwise.
    """
    first_idx, last_idx = indices
    # If first index is 0, it means there are no auctions or no intraday data
    if first_idx == 0:
        return False

    # If the last index is the last index of the dataframe, it means there are no auctions
    if last_idx == df.index[-1]:
        return False

    return True

def get_auction_data(df: pd.DataFrame, indices: tuple[int, int]) -> pd.DataFrame:
    """
    Get the auction data from the trade data.

    Parameters
    ----------
    df : DataFrame containing the trade data.
    indices : Indices of the first and last intraday data.
    date : Date of the data to be processed.

    Returns
    -------
    DataFrame containing the aggregated trade data.
    """

    first_idx, last_idx = indices

    # Last opening auction index is before the first intraday index
    opening_auction = df.iloc[:first_idx]

    # First closing auction index is after the last intraday index
    closing_auction = df.iloc[last_idx + 1:]

    # Checking if there are any rows that are not an auction
    opening_all_auction = (opening_auction['flag'] == 'AUCTION').all()
    closing_all_auction = (closing_auction['flag'] == 'AUCTION') .all()
    if not opening_all_auction or not closing_all_auction:
        return pd.DataFrame()

    # Merge the opening and closing auction data
    opening_price, opening_quantity = merge_auction_data(opening_auction)
    closing_price, closing_quantity = merge_auction_data(closing_auction)

    df = pd.DataFrame(columns=['Price', 'Quantity'])
    df.loc['open'] = [opening_price, opening_quantity]
    df.loc['close'] = [closing_price, closing_quantity]

    return df

def merge_auction_data(auction_rows: pd.DataFrame) -> tuple[float, int]:
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
    quantity = auction_rows['quantity'].sum()
    return price, quantity

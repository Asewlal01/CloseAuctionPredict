from DataProcessing.Processors import TradeProcessor, LimitOrderBookProcessor
import pandas as pd
import multiprocessing
import os

datatype = dict[str, pd.DataFrame]


class FileMerger:
    def __init__(self, trade_file_path: str, lob_file_path: str):
        """
        Initialize the FileMerger with trade and limit order book processors.

        Parameters
        ----------
        trade_file_path : Path to the trade file to be processed.
        lob_file_path : Path to the limit order book file to be processed.
        """
        self.lob_processor = LimitOrderBookProcessor(lob_file_path)
        self.trade_processor = TradeProcessor(trade_file_path, self.lob_processor)

        symbol = lob_file_path.split('_')[-1]
        symbol = symbol.split('.')[0]
        self.symbol = symbol

    def combine_datasets(self, save_path: str) -> None:
        """
        Combine the trade and limit order book data into a single dataframe.
        The data is aggregated into 1 minute intervals.
        """
        trade_data = self.trade_processor.aggregated_data
        lob_data = self.lob_processor.aggregated_data

        # Create the save path
        save_path = os.path.join(save_path, self.symbol)
        os.makedirs(save_path, exist_ok=True)

        combine_all_datasets(trade_data, lob_data, save_path)


def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    The aggregated trade dataset does not handle missing values, so we need to fix them after combining the datasets.
    The VWAP price is forward filled, and it assumed that the other values are the same as the current VWAP price.
    Empty volume means no trade, hence it is set to 0.

    Parameters
    ----------
    df : DataFrame to fix.

    Returns
    -------
    DataFrame containing the fixed dataset.
    """

    # Forward filling the VWAP price
    df['vwap'] = df['vwap'].ffill()

    # Filling the other values with the current VWAP price
    df['open'] = df['open'].fillna(df['vwap'])
    df['high'] = df['high'].fillna(df['vwap'])
    df['low'] = df['low'].fillna(df['vwap'])
    df['close'] = df['close'].fillna(df['vwap'])

    # No trades so no traded volume
    df['quantity'] = df['quantity'].fillna(0)

    return df

def combine_dataset(trade_df, lob_df, save_path) -> None:
    """
    Combine trade and limit order book data into a single dataframe.

    Parameters
    ----------
    trade_df : DataFrame of aggregated trades.
    lob_df : DataFrame of aggregated limit order books.
    save_path : Path to save the combined data.
    """
    # Merge the two dataframes on the 't' column
    # Check if unique indices
    if not trade_df.index.is_unique:
        # Print the indices that are not unique
        raise ValueError("Trade dataframe has non-unique indices")

    if not lob_df.index.is_unique:
        raise ValueError("LOB dataframe has non-unique indices")

    lob_df.index = lob_df.index.strftime('%H:%M:%S:%f')
    merged_df = pd.concat([lob_df, trade_df], axis=1)

    indices = merged_df.index.tolist()[:-2]
    indices = ['opening'] + indices + ['closing']
    merged_df = merged_df.loc[indices]

    merged_df = fix_missing(merged_df)

    # Check if there are still nans
    if merged_df.iloc[1:-1].isna().any(axis=1).any():
        raise ValueError("NaNs remain in the merged dataframe")

    # Save the combined dataframe to a parquet file
    merged_df.to_parquet(save_path, index=False)


def unpack_args(args):
    """
    Unpack the arguments for the combine_dataset function.

    Parameters
    ----------
    args : Tuple containing the trade and limit order book dataframes.

    Returns
    -------
    Tuple containing the trade and limit order book dataframes.
    """
    return combine_dataset(*args)

def combine_all_datasets(trade_data: datatype, lob_data: datatype, save_path: str) -> None:
    """
    Combine trade and limit order book data into a single dataframe.

    Parameters
    ----------
    trade_data : Data of aggregated trades.
    lob_data : Data of aggregated limit order books.
    save_path : Path to save the combined data.

    Returns
    -------
    Dictionary with each date as the key and the combined data as the value.
    """
    # Get the dates that are common in both dataframes
    trade_keys = set(trade_data.keys())
    lob_keys = set(lob_data.keys())
    common_dates = trade_keys.intersection(lob_keys)

    n_cores = multiprocessing.cpu_count()
    items = [
        (trade_data[date], lob_data[date], os.path.join(save_path, f'{date}.parquet')) for date in common_dates
    ]
    with multiprocessing.Pool(n_cores) as pool:
        # Use multiprocessing to process each group in parallel
        pool.map(unpack_args, items)
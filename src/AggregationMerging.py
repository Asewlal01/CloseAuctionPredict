import pandas as pd
import os


def merge_trades(path_to_stock):
    """
    Merge trades using all the files in the folder

    Parameters
    ----------
    path_to_stock : Path to stock trades

    Returns
    -------

    """
    files = os.listdir(f'{path_to_stock}')

    if len(files) == 0:
        print(f'Current folder has no files')
        return

    # Storing each price and volume
    prices = []
    volumes = []

    # Going through each file
    for file in files:
        df = pd.read_parquet(f'{path_to_stock}/{file}')

        prices.append(df['price'].values)
        volumes.append(df['quantity'].values)

    # Convert to dataframe
    df_prices = pd.DataFrame(prices).T
    df_volumes = pd.DataFrame(volumes).T

    return df_prices, df_volumes

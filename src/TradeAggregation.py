import pandas as pd
import gzip
import numpy as np
import os

def load_file(file):
    """
    Load data from a zipped file. Data within file is assumed to be from 2021, and contain 4 columns:
    time, price, quantity and flag.

    :param file: Path to the file
    :return: Dataframe with the data
    """
    with gzip.open(file, 'rb') as f:
        dataframe = pd.read_csv(f, names=['time', 'price', 'quantity', 'flag'], skiprows=1)

    # Convert time to actual time
    try:
        dataframe['time'] = pd.to_datetime(dataframe['time'])
    # In some files the time is not in the correct format hence we need the mixed format to avoid errors
    except:
        dataframe['time'] = pd.to_datetime(dataframe['time'], format='mixed')

    # Remove rows which are not in 2021
    rows_to_include = dataframe['time'].dt.year == 2021
    dataframe = dataframe[rows_to_include]

    return dataframe

def file_to_name(file, stocks_mapping):
    """
    Get the name of the stock from the file name.

    :param file: Path to the file
    :param stocks_mapping: Mapping from id to name with syd as the key
    :return: Name of the stock
    """
    # Get numeric representation of stock
    syd = file.split('_')[-1]

    # Remove file extension
    syd = syd.split('.')[0]

    # Get name of stock
    row = stocks_mapping[stocks_mapping['syd'] == int(syd)]
    name = row['name'].iloc[0]

    # Replace space with underscores
    name = name.replace(' ', '_')

    return name

def get_auction_data(auction_rows):
    """
    Get the opening and closing price and volume from the auction data.

    :param auction_rows: Rows with flag 'AUCTION'
    :return: Opening and closing price and volume as a tuple
    """
    # Get the rows with the opening and closing price
    unique_prices = auction_rows['price'].unique()

    # Making sure that there are only two unique prices
    if len(unique_prices) != 2:
        return None

    # Get the rows with the opening and closing price
    opening_rows = auction_rows[auction_rows['price'] == unique_prices[0]]
    closing_rows = auction_rows[auction_rows['price'] == unique_prices[-1]]

    # Get the auction price and volume
    opening = [opening_rows['price'].iloc[0], opening_rows['quantity'].sum()]
    closing = [closing_rows['price'].iloc[0], closing_rows['quantity'].sum()]

    # Converting to a dataframe
    auction_df = pd.DataFrame([opening, closing], columns=['price', 'quantity'], index=['opening', 'closing'])

    return auction_df


def volume_weighted_average_price(dataframe):
    """
    Calculate the volume weighted average price of the data.

    :param dataframe: Dataframe with columns 'price' and 'quantity'
    :return: Volume weighted average price of the dataframe
    """
    if dataframe['quantity'].sum() == 0:
        return np.nan

    # Calculate the volume weighted average price
    vwap = (dataframe['price'] * dataframe['quantity']).sum() / dataframe['quantity'].sum()

    return vwap

def aggregate_trades(data_day, interval='1min'):
    """
    Aggregate the all the trades within a day to a given interval.

    :param data_day: Dataframe with the data for a day
    :param interval: Resampling interval
    :return:
    """
    # Getting the indices of the auction rows and normal rows
    auction_flagged_rows = data_day['flag'] == 'AUCTION'
    normal_flagged_rows = data_day['flag'] == 'NORMAL'

    # Get the auction and normal rows
    auction_rows = data_day[auction_flagged_rows]
    normal_rows = data_day[normal_flagged_rows]

    # Obtaining the opening and closing prices
    auction_df = get_auction_data(auction_rows)
    if auction_df is None:
        return None

    # Make sure we start at 08:00:00 and end at 16:30:00
    day = normal_rows['time'].dt.date.iloc[0]
    tz = normal_rows['time'].dt.tz
    start = pd.to_datetime(f'{day} 08:00:00').tz_localize(tz)
    end = pd.to_datetime(f'{day} 16:30:00').tz_localize(tz)
    normal_rows = normal_rows[(normal_rows['time'] >= start) & (normal_rows['time'] <= end)]

    resample = normal_rows.resample(interval, on='time', label='left')
    df = pd.DataFrame(normal_rows['time']).resample(interval, on='time', label='left').count()
    df['price'] = resample.apply(volume_weighted_average_price)
    df['quantity'] = resample['quantity'].sum()

    # Make sure that all the intervals are present
    time_index = pd.date_range(start=start, end=end, freq=interval)
    df = df.reindex(time_index)

    # Converting the index to a string with only the hours, minutes and seconds
    df.index = df.index.strftime('%H:%M:%S')

    # Remove row at 16:30:00
    df = df.iloc[:-1]

    # Add the opening and closing prices
    df = pd.concat([auction_df.loc[['opening']], df])
    df = pd.concat([df, auction_df.loc[['closing']]])

    # # Set missing values to previous value for price and 0 for quantity
    df['price'] = df['price'].ffill()
    df['quantity'] = df['quantity'].fillna(0)

    return df

def aggregate_trades_in_file(file, stocks_mapping, path_to_save, interval='1min', days_to_save=np.inf,
                             method_to_save='parquet', verbose=False):
    """
    Aggregate all the days within a file to a given interval.

    :param file: Path to the file
    :param stocks_mapping: Mapping from id to name with syd as the key
    :param path_to_save: Path to save the aggregated data
    :param interval: Resampling interval
    :param days_to_save: Number of days to save. If -1, all days are saved
    :param method_to_save: Method to save the data. Either 'parquet' or 'csv'
    :param verbose: If True, print progress
    :return: Dataframe with the aggregated trades
    """
    # Load the data and make sure that we have data
    data = load_file(file)
    stock_name = file_to_name(file, stocks_mapping)
    if data.empty:
        print(f'No data in file: {file}/{stock_name}')
        return None

    # Make sure path exists
    if not os.path.exists(path_to_save):
        print(f'Creating directory: {path_to_save}')
        os.makedirs(path_to_save)

    # Group by days
    days = data['time'].dt.date
    day_groups = data.groupby(days)

    # Creating folder to save all the days
    if not os.path.exists(f'{path_to_save}/{stock_name}'):
        os.makedirs(f'{path_to_save}/{stock_name}')

    days_processed = 0
    for name, group in day_groups:
        # Aggregate the trades and skip if there is no data
        df = aggregate_trades(group, interval)
        if df is None:
            continue

        # Save the data as parquet
        match method_to_save:
            case 'parquet':
                df.to_parquet(f'{path_to_save}/{stock_name}/{name}.parquet')
            case 'csv':
                df.to_csv(f'{path_to_save}/{stock_name}/{name}.csv', index=False)

        # Show current progress
        if verbose:
            print(f'Saved trades of {stock_name} at {name} to {method_to_save}')

        # Increment and stop if we have processed the desired number of days
        days_processed += 1
        if days_processed >= days_to_save:
            break


    return stock_name
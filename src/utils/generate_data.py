import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

def generate_trade_day(start_time: datetime, expected_trades: int, expected_price: float,
                       price_std: float, expected_volume: float, volume_std: float) -> pd.DataFrame:
    """
    Generate a single day's worth of trade data. Assumed to be between 8 hours of trading time.

    Parameters
    ----------
    start_time : Start time of the day
    expected_trades : Expected number of trades for the day
    expected_price : Expected average price of trades
    price_std : Standard deviation of trade prices
    expected_volume : Expected average volume of trades
    volume_std : Standard deviation of trade volumes

    Returns
    -------
    DataFrame with trade data for the day.
    """
    # Generate the number of trades
    n_trades = np.random.poisson(expected_trades)

    # Generate timestamps
    total_seconds = 2 * 60 * 60  # 2 hours in seconds
    timestamps = [start_time + timedelta(seconds=np.random.randint(0, total_seconds)) for _ in range(n_trades)]
    timestamps.sort()

    # Generate trade prices
    price_delta = np.random.normal(0, price_std, n_trades)
    volume_delta = np.random.normal(0, volume_std, n_trades)

    # Replace negative with zero
    prices = np.maximum(expected_price + price_delta, 0)
    volumes = np.maximum(expected_volume + volume_delta, 0)

    # Replace zero with expected values
    prices[prices == 0] = expected_price
    volumes[volumes == 0] = expected_volume

    # Round to 2 decimal places and volume as integers since they represent shares
    prices = np.round(prices, 2)
    volumes = volumes.astype(int)

    # Create DataFrame
    trades = pd.DataFrame({
        't': timestamps,
        'price': prices,
        'quantity': volumes,
        'flag': ['NORMAL'] * n_trades
    })

    # Replace first and last row by auctions
    trades.iloc[0, trades.columns.get_loc('flag')] = 'AUCTION'
    trades.iloc[-1, trades.columns.get_loc('flag')] = 'AUCTION'

    return trades

def generate_trades(start_time: datetime, n_days: int, expected_trades: int, expected_price: float,
                       price_std: float, expected_volume: float, volume_std: float) -> pd.DataFrame:
    """
    Generate trades for multiple days.

    Parameters
    ----------
    start_time : Start time of the day
    n_days : Number of days to generate trades for
    expected_trades : Expected number of trades for the day
    expected_price : Expected average price of trades
    price_std : Standard deviation of trade prices
    expected_volume : Expected average volume of trades
    volume_std : Standard deviation of trade volumes

    Returns
    -------
    DataFrame with trade data for multiple days.
    """
    all_trades = []
    for i in range(n_days):
        start_time = start_time + timedelta(days=1)
        trades = generate_trade_day(start_time, expected_trades, expected_price, price_std, expected_volume, volume_std)
        all_trades.append(trades)

    # Concatenate all trades into a single DataFrame
    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # Ensure the 't' column is datetime
    all_trades_df['t'] = pd.to_datetime(all_trades_df['t'])

    # Localize to UTC if naive (i.e., no timezone info)
    if all_trades_df['t'].dt.tz is None:
        all_trades_df['t'] = all_trades_df['t'].dt.tz_localize(timezone.utc)

    # Convert to ISO 8601 format with space separator and microseconds
    all_trades_df['t'] = all_trades_df['t'].apply(lambda x: x.isoformat(sep=' ', timespec='microseconds'))

    return all_trades_df


def generate_quote_day(start_time: datetime, end_time: datetime, expected_price: float,
                       price_std: float, expected_volume: float, volume_std: float) -> pd.DataFrame:
    """
    Generate a single day's worth of quote data. Assumed to be between 8 hours of trading time. All observations
    are at 1 minute intervals, with random jitter of ±100ms.

    Parameters
    ----------
    start_time : Start time of the day
    end_time : End time of the day
    expected_price : Expected average price of quotes
    price_std : Standard deviation of quote prices
    expected_volume : Expected average volume of quotes
    volume_std : Standard deviation of quote volumes

    Returns
    -------
    DataFrame with quote data for the day.
    """
    quote_rows = []
    current_time = start_time
    n_levels = 5

    # Keep generating quotes until we reach the end time
    while current_time < end_time:
        # Add random jitter of ±100ms
        jitter = timedelta(milliseconds=np.random.randint(-100, 100))
        t = current_time + jitter

        # Everything is around mid price
        mid_price = expected_price + np.random.normal(0, price_std)

        bid_prices = [np.round(mid_price - price_std * (i + 1), 2) for i in range(n_levels)]
        ask_prices = [np.round(mid_price + price_std * (i + 1), 2) for i in range(n_levels)]

        bid_sizes = expected_volume + np.random.normal(0, volume_std, size=n_levels)
        ask_sizes = expected_volume + np.random.normal(0, volume_std, size=n_levels)

        row = {'t': t, 'nature': 'q'}
        for i in range(n_levels):
            row[f'bb{i+1}'] = bid_prices[i]
            row[f'bbvol{i+1}'] = bid_sizes[i]
            row[f'ba{i+1}'] = ask_prices[i]
            row[f'bavol{i+1}'] = ask_sizes[i]

        quote_rows.append(row)
        current_time += timedelta(minutes=1)

    return pd.DataFrame(quote_rows)

def generate_quotes(start_time: datetime, end_time: datetime, n_days: int, expected_price: float,
                       price_std: float, expected_volume: float, volume_std: float) -> pd.DataFrame:
    """
    Generate quotes for multiple days.

    Parameters
    ----------
    start_time : Start time of the day
    end_time : End time of the day
    n_days : Number of days to generate quotes for
    expected_price : Expected average price of quotes
    price_std : Standard deviation of quote prices
    expected_volume : Expected average volume of quotes
    volume_std : Standard deviation of quote volumes

    Returns
    -------
    DataFrame with quote data for multiple days.
    """
    all_quotes = []
    for i in range(n_days):
        start_time = start_time + timedelta(days=1)
        end_time = end_time + timedelta(days=1)
        quotes = generate_quote_day(start_time, end_time, expected_price, price_std, expected_volume, volume_std)
        all_quotes.append(quotes)

    # Concatenate all trades into a single DataFrame
    all_quotes_df = pd.concat(all_quotes, ignore_index=True)

    # Step 1: Ensure the 't' column is datetime
    all_quotes_df['t'] = pd.to_datetime(all_quotes_df['t'])

    # Step 2: Localize to UTC if naive (i.e., no timezone info)
    if all_quotes_df['t'].dt.tz is None:
        all_quotes_df['t'] = all_quotes_df['t'].dt.tz_localize(timezone.utc)

    # Step 3: Convert to ISO 8601 format with space separator and microseconds
    all_quotes_df['t'] = all_quotes_df['t'].apply(lambda x: x.isoformat(sep=' ', timespec='microseconds'))

    return all_quotes_df

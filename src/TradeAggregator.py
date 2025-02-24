import pandas as pd
import numpy as np


class TradeAggregator:
    def __init__(self, exchange, df):
        """
        This class is used to aggregate trade data into 1 minute intervals. It also moves the time of the trade data to the
        opening time of the exchange.

        Parameters
        ----------
        exchange : Code of the exchange
        df : Dataframe with the trade data

        """
        self.exchange = exchange
        self.df = df

        # Get the opening and closing time of the exchange
        self.opening = None
        self.closing = None
        self.exchange_times()

    def exchange_times(self):
        """
        This function sets the opening and closing time of the exchange based on the exchange code.

        Returns
        -------
        Tuple with the opening and closing time of the exchange
        """

        # Finding the row that corresponds to the exchange
        exchange_row = self.df[self.df['code'] == self.exchange]

        # Get opening and closing time
        opening = exchange_row['Opening'].values[0]
        opening = pd.to_datetime(opening, format='%H:%M:%S').time()
        self.opening = opening

        closing = exchange_row['Closing'].values[0]
        closing = pd.to_datetime(closing, format='%H:%M:%S').time()
        self.closing = closing

    def move_to_opening(self, df):
        """
        This function increments the time of the dataframe to match the opening time. Exchanges denote their opening times in a
        timezone which may be different from the timezone of the data.

        Parameters
        ----------
        df : Dataframe with the trade data

        Returns
        -------
        Dataframe with the time incremented to the opening time

        """
        # We only need the hour of the opening time to increment the time
        opening_hour = self.opening.hour
        first_trade_hour = df.iloc[0]['time'].hour
        increment = first_trade_hour - opening_hour
        df['time'] = df['time'] - pd.Timedelta(hours=increment)

        return df

    def auction_normal_split(self, df):
        """
        This function splits the trade data into auction and normal trades.

        Parameters
        ----------
        df : Dataframe with the trade data

        Returns
        -------
        Tuple with the auction and normal trade dataframes
        """
        auction = df[df['flag'] == 'AUCTION'].copy()
        normal = df[df['flag'] == 'NORMAL'].copy()

        return auction, normal

    def get_auction_data(self, auction_df):
        """
        This function returns the closing and auction price and quantities based on the auction trade data.

        Parameters
        ----------
        auction_df : Dataframe with the auction trades

        Returns
        -------
        Dataframe with the closing and auction prices and quantities
        """
        # Empty df
        if auction_df.empty:
            return None

        opening = auction_df.iloc[0]
        closing = auction_df.iloc[-1]

        # On normal days the closing auction time is after the closing time
        if closing['time'].time() < self.closing:
            return None

        # If the opening and closing price are the same then one of the prices is missing
        if opening['price'] == closing['price']:
            print('Missing opening or closing price')
            return None

        # Get the auction price and quantity
        opening_price = opening['price']
        opening_rows = auction_df[auction_df['price'] == opening_price]
        opening_quantity = np.sum(opening_rows['quantity'])

        closing_price = closing['price']
        closing_rows = auction_df[auction_df['price'] == closing_price]
        closing_quantity = np.sum(closing_rows['quantity'])

        df = pd.DataFrame(columns=['price', 'quantity'])
        df.loc['opening'] = [opening_price, opening_quantity]
        df.loc['closing'] = [closing_price, closing_quantity]

        return df

    def add_exchange_times(self, normal_df):
        """
        This function adds the opening and closing time of the exchange to the dataframe.

        Parameters
        ----------
        normal_df : Dataframe of the trade data with only the normal trades

        Returns
        -------
        Dataframe with the opening and closing time of the exchange added
        """

        day = normal_df['time'].dt.date.iloc[0]
        tz = normal_df['time'].dt.tz

        # Last trades actually happen one second before the closing time
        opening_day = pd.Timestamp(f'{day} {self.opening}', tz=tz)
        closing_day = pd.Timestamp(f'{day} {self.closing}', tz=tz) - pd.Timedelta(seconds=1)

        opening_index = normal_df.index[0] - 1
        closing_index = normal_df.index[-1] + 1

        normal_df.loc[opening_index] = [opening_day, np.nan, np.nan, 'NORMAL']
        normal_df.loc[closing_index] = [closing_day, np.nan, np.nan, 'NORMAL']
        normal_df = normal_df.sort_index()

        return normal_df

    def resample(self, normal_df, freq='1min'):
        """
        This function resamples the trade data to the specified frequency.

        Parameters
        ----------
        normal_df : Dataframe with the normal trade data
        freq : Resampling frequency

        Returns
        -------
        Dataframe with the resampled data
        """
        resample = normal_df.resample(freq, on='time', label='left')
        resampled_df = resample.apply(aggregate)

        # For NaN values it is assumed that the price did not change and volume was 0
        resampled_df['price'] = resampled_df['price'].ffill()
        resampled_df['quantity'] = resampled_df['quantity'].fillna(0)

        # Removing the day from the index and converting the time to a string
        resampled_df.index = resampled_df.index.strftime('%H:%M:%S')

        return resampled_df

    def process_df(self, df):
        """
        This function processes all the trade data for a single day. It returns a df with the resampled data and the auction data.

        Parameters
        ----------
        df : Dataframe with the trade data

        Returns
        -------
        Dataframe with the resampled data and the auction data
        """

        df = self.move_to_opening(df)
        auction, normal = self.auction_normal_split(df)
        auction_df = self.get_auction_data(auction)

        if auction_df is None:
            return None

        normal = self.add_exchange_times(normal)
        resampled = self.resample(normal)

        # This sets the opening auction at the beginning and closing auction at the end of the resampled data
        resampled = pd.concat([
            auction_df.iloc[[0]],
            resampled,
            auction_df.iloc[[1]]
        ])

        return resampled

def aggregate(df):
    """
    This function aggregates the trades and quantities within a dataframe to a single row. This row contains the volume weighted average price (VWAP) and the total quantity traded.

    Parameters
    ----------
    df : Dataframe to be aggregated. Assumed to contain a row for each trade with columns 'price' and 'quantity'.

    Returns
    -------
    Pandas series with the aggregated values
    """
    # No point in aggregating if the dataframe is empty
    if df['quantity'].sum() == 0:
        return pd.Series({'price': np.nan, 'quantity': np.nan})

    quantity = np.sum(df['quantity'])
    price = np.sum(df['price'] * df['quantity']) / quantity

    return pd.Series({'price': price, 'quantity': quantity})
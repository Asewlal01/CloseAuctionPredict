import pandas as pd
import os
from src.data_processing.TradeAggregator import TradeAggregator

trades_path = '../data/raw/trades'
exchanges = [exchange for exchange in os.listdir(trades_path) if os.path.isdir(f'{trades_path}/{exchange}')][1:]
df_exchanges = pd.read_csv('../data/raw/exchanges.csv', keep_default_na=False)

aggregated_trades = '../data/processed/aggregated_trades'

for i, exchange in enumerate(exchanges):
    print(f'Processing {exchange}')
    exchange_path = f'{trades_path}/{exchange}'
    aggregator = TradeAggregator(exchange, df_exchanges)

    stocks = os.listdir(exchange_path)
    for stock in stocks:
        stock_path = f'{exchange_path}/{stock}'
        days = os.listdir(stock_path)

        for day in days:
            df = pd.read_parquet(f'{stock_path}/{day}')
            resampled_df = aggregator.process_df(df)

            if resampled_df is None:
                continue

            save_path = f'{aggregated_trades}/{stock}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            day = day.split('.')[0]
            resampled_df.to_parquet(f'{save_path}/{day}.parquet')


    print(f'{i+1}/{len(exchanges)} exchanges processed')
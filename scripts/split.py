from src.TradeAggregation import load_file
import os
import pandas as pd

def split_per_day(df, save_path):
    grouped = df.groupby(df['time'].dt.date)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for day, data in grouped:
        data.to_parquet(f'{save_path}/{day}.parquet')

if __name__ == '__main__':
    # Raw trades
    trade_path = '../data/raw/trades'
    exchange_folders = [folder for folder in os.listdir(trade_path) if os.path.isdir(f'{trade_path}/{folder}')]

    for exchange in exchange_folders:
        exchange_path = f'{trade_path}/{exchange}'
        stocks = [stock for stock in os.listdir(exchange_path) if stock.endswith('.gz')]

        for stock in stocks:
            stock_path = f'{exchange_path}/{stock}'
            df = load_file(stock_path)
            name = stock.split('.')[0]
            save_path = f'{exchange_path}/{name}'

            split_per_day(df, save_path)
            # Deleting the file
            os.remove(stock_path)
            print(f'Split {name}')

        print(f'{exchange} done \n')


from src.TradeAggregation import *
import pandas as pd
import os

# Path to files
trades_path = '../data/raw/trades'
path_to_save = '../data/processed/aggregated_trades'

# Mapping
stock_mapping = pd.read_csv('../data/stocks_mapping.csv')

# Processing
for file in os.listdir(trades_path):
    stock_name = aggregate_trades_in_file(f'{trades_path}/{file}', stock_mapping, path_to_save, interval='1min',
                                          days_to_save=30, method_to_save='parquet', verbose=False)

    if stock_name is not None:
        print(f'Processed trades of {stock_name}')
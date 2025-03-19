import os
import pandas as pd
from DataProcessing.TradeAggregator import TradeAggregator
from tqdm import tqdm

trade_dir = 'data/processed/split_trades'
save_dir = 'data/processed/aggregated_trades'
exchange_data = pd.read_csv('data/raw/exchanges.csv', keep_default_na=False)
aggregation_interval = '1min'

n_stocks = sum([len(os.listdir(os.path.join(trade_dir, exchange))) for exchange in os.listdir(trade_dir)])
print(f'Total number of stocks: {n_stocks}')

with tqdm(total=n_stocks) as pbar:
    for exchange in os.listdir(trade_dir):
        exchange_info = exchange_data[exchange_data['bb'] == exchange]
        if exchange_info.empty:
            print(f'No exchange info for {exchange}')
            continue

        exchange_path = os.path.join(trade_dir, exchange)
        aggregator = TradeAggregator(save_dir, exchange_info, aggregation_interval)

        for stock in os.listdir(exchange_path):
            stock_path = os.path.join(exchange_path, stock)
            aggregator.aggregate_stock(stock_path)
            pbar.update(1)


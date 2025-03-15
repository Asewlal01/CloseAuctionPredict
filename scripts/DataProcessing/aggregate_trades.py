import os
import pandas as pd
from DataProcessing.TradeAggregator import TradeAggregator
from tqdm import tqdm

trade_dir = 'data/processed/split_trades'
save_dir = 'data/processed/aggregated_trades'
exchange_data = pd.read_csv('data/raw/exchanges.csv', keep_default_na=False)
aggregation_interval = '1min'

for i, exchange in enumerate(os.listdir(trade_dir)):
    exchange_info = exchange_data[exchange_data['bb'] == exchange]
    if exchange_info.empty:
        print(f'No exchange info for {exchange}')
        continue

    exchange_path = os.path.join(trade_dir, exchange)
    aggregator = TradeAggregator(save_dir, exchange_info, aggregation_interval)

    for stock in tqdm(os.listdir(exchange_path)):
        stock_path = os.path.join(exchange_path, stock)
        aggregator.aggregate_stock(stock_path)

    print(f'Finished {(i+1)/len(os.listdir(trade_dir)) * 100:.2f}% of exchanges')



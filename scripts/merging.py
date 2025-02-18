import os
from src.AggregationMerging import merge_trades

# Path to trade files
processed_trades = '../data/processed/aggregated_trades'
save_path = '../data/processed/merged_trades'

# Go through each stock
for stock in os.listdir(processed_trades):
    df_prices, df_volumes = merge_trades(f'{processed_trades}/{stock}')

    # Creating folder to save
    if not os.path.exists(f'{save_path}/{stock}'):
        os.makedirs(f'{save_path}/{stock}')

    # Saving as parquet
    df_prices.to_parquet(f'{save_path}/{stock}/{stock}_prices.parquet')
    df_volumes.to_parquet(f'{save_path}/{stock}/{stock}_volumes.parquet')
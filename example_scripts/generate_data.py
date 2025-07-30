from datetime import datetime
from utils.generate_data import *
import os

# Starting at 9 AM and ending at 11 AM, for 10 days
start_time = datetime(2023, 1, 1, 9, 0, 0)
end_time = datetime(2023, 1, 1, 11, 0, 0)
days = 100   # Number of days to generate data for

# Where to save the data
save_path = 'example_data/raw_data'
os.makedirs(f'{save_path}/trade_data', exist_ok=True)
os.makedirs(f'{save_path}/quote_data', exist_ok=True)

# 10 different stocks with index -> stock_id mapping
stock_params = {
    1001: {'expected_trades': 300, 'expected_price': 180.00, 'price_std': 0.8,  'expected_volume': 400, 'volume_std': 50},
    1002: {'expected_trades': 250, 'expected_price': 320.00, 'price_std': 1.2,  'expected_volume': 500, 'volume_std': 80},
    1003: {'expected_trades': 200, 'expected_price': 2800.00, 'price_std': 5.0, 'expected_volume': 200, 'volume_std': 30},
    1004: {'expected_trades': 220, 'expected_price': 135.00, 'price_std': 1.0,  'expected_volume': 300, 'volume_std': 60},
    1005: {'expected_trades': 180, 'expected_price': 350.00, 'price_std': 2.0,  'expected_volume': 250, 'volume_std': 40},
    1006: {'expected_trades': 260, 'expected_price': 700.00, 'price_std': 10.0, 'expected_volume': 600, 'volume_std': 100},
    1007: {'expected_trades': 150, 'expected_price': 450.00, 'price_std': 3.5,  'expected_volume': 180, 'volume_std': 25},
    1008: {'expected_trades': 280, 'expected_price': 220.00, 'price_std': 1.5,  'expected_volume': 350, 'volume_std': 70},
    1009: {'expected_trades': 240, 'expected_price': 95.00,  'price_std': 0.6,  'expected_volume': 420, 'volume_std': 55},
    1010: {'expected_trades': 210, 'expected_price': 510.00, 'price_std': 4.0,  'expected_volume': 300, 'volume_std': 45},
}

for stock, values in stock_params.items():
    # Generate trade data for each stock
    trade_df = generate_trades(
        start_time=start_time,
        n_days=days,
        expected_trades=values['expected_trades'],
        expected_price=values['expected_price'],
        price_std=values['price_std'],
        expected_volume=values['expected_volume'],
        volume_std=values['volume_std']
    )

    # Generate quote data for each stock
    quote_df = generate_quotes(
        start_time=start_time,
        end_time=end_time,
        n_days=days,
        expected_price=values['expected_price'],
        price_std=values['price_std'],
        expected_volume=values['expected_volume'],
        volume_std=values['volume_std']
    )

    # Saving to compressed csv files
    trade_df.to_csv(f'{save_path}/trade_data/stock_{stock}.gz', index=False, compression='gzip')
    quote_df.to_csv(f'{save_path}/quote_data/stock_{stock}.gz', index=False, compression='gzip')
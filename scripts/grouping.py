import os
import pandas as pd

def get_stock_info(stocks, syd):
    # Find row
    row = stocks[stocks['syd'] == syd]

    # Check if row exists
    if row.empty:
        raise ValueError(f"Stock {syd} not found in stocks.csv")

    # Getting the exchange
    exchange = row['bb'].values[0].split(' ')[1]

    # The name
    name = row['name'].values[0]

    return exchange, name

def get_syd(file):
    # Getting the syd with file extension
    syd = file.split('_')[3]

    # Removing the file extension
    syd = syd.split('.')[0]

    return int(syd)


if __name__ == '__main__':
    stocks = pd.read_csv('../data/raw/stocks.csv')
    trade_path = '../data/raw/trades'
    for file in os.listdir(trade_path):
        syd = get_syd(file)
        exchange, name = get_stock_info(stocks, syd)

        # Create folder for exchange
        exchange_path = os.path.join(trade_path, exchange)
        if not os.path.exists(exchange_path):
            os.makedirs(exchange_path)

        # Move to folder
        name = name.replace(' ', '_')
        os.rename(os.path.join(trade_path, file), os.path.join(exchange_path, f'{name}.gz'))
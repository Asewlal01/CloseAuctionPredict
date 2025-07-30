from DataProcessing.Processors.TradeProcessor import TradeProcessor
import os
from tqdm import tqdm


lob_path = 'example_data/processed/quote_data'  # Processed limit order book data path
trade_path = 'example_data/raw_data/trade_data' # Raw trade data path
save_path = 'example_data/processed/trade_data' # Processed trade data save path

def get_index(stock_path):
    """Extracts the index from the stock file name."""
    idx = stock_path.split('_')[-1]
    idx = idx.split('.')[0]
    return idx

# Maximum price for the stocks (All days with prices above this are removed)
max_price = 1e6

stocks = sorted(os.listdir(trade_path))
for stock in tqdm(stocks):
    # Get LOB path using index from stock name
    idx = get_index(stock)
    stock_trade_path = os.path.join(trade_path, stock)
    stock_lob_path = os.path.join(lob_path, idx)

    # Process the file
    processor = TradeProcessor(stock_trade_path, stock_lob_path, max_price)
    processor.process_file()

    # Save the results
    stock_save_path = os.path.join(save_path, str(idx))
    processor.save_results(stock_save_path)


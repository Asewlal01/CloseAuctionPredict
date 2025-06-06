from DataProcessing.Processors.LimitOrderBookProcessor import LimitOrderBookProcessor
import os
from tqdm import tqdm

# Path to the lob data
lob_path = 'data/raw/limit_order_book_data'
stocks = os.listdir(lob_path)
stocks.sort()

# Path to save the processed data
save_path = 'data/processed/limit_order_book'

# Maximum price for the stocks
max_price = 1e6

for stock in tqdm(stocks):
    # Get the path to the file
    stock_path = os.path.join(lob_path, stock)

    # Process the file
    processor = LimitOrderBookProcessor(stock_path, max_price)
    processor.process_file()

    # Create the save path
    idx = processor.get_id()
    stock_save_path = os.path.join(save_path, idx)

    # Saving the results
    processor.save_results(stock_save_path)
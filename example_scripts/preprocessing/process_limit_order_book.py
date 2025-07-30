from DataProcessing.Processors.LimitOrderBookProcessor import LimitOrderBookProcessor
import os
from tqdm import tqdm


# Path to the lob data
lob_path = 'example_data/raw_data/quote_data'

# Path to save the processed data
save_path = 'example_data/processed/quote_data'

# Maximum price for the stocks (All days with prices above this are removed)
max_price = 1e6

stocks = sorted(os.listdir(lob_path))
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
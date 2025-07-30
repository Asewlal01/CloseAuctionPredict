from DataProcessing.Processors.AuctionProcessor import AuctionProcessor
import os
from tqdm import tqdm


trade_path = 'example_data/raw_data/trade_data' # Raw auction data path
save_path = 'example_data/processed/auction_data' # Processed auction data save path

stocks = sorted(os.listdir(trade_path))
for stock in tqdm(stocks):
    # Get the path to the file
    stock_path = os.path.join(trade_path, stock)

    # Process the file
    processor = AuctionProcessor(stock_path)
    processor.process_file()

    # Save the results
    idx = processor.get_id()
    stock_save_path = os.path.join(save_path, idx)
    processor.save_results(stock_save_path)
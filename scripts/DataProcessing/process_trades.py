from DataProcessing.FileMerger import FileMerger
import os
from tqdm import tqdm

path_to_trades = '/home/amish/Projects/CloseAuctionPredict/data/raw_trade_data'
path_to_lob = '/home/amish/Projects/CloseAuctionPredict/data/raw_quote_data'
save_path = '/home/amish/Projects/CloseAuctionPredict/data/aggregated_files'
files = os.listdir(path_to_trades)
files.sort()

start_at = 27 + 177
files = files[start_at:]

for file in tqdm(files):
    if not file.endswith('.gz'):
        continue

    extension = file.split('/')[-1].split('data_')[1]
    lob_file = os.path.join(path_to_lob, f'1sec_quote_data_{extension}')
    trade_file = os.path.join(path_to_trades, file)

    try:
        file_merger = FileMerger(trade_file, lob_file)
        file_merger.combine_datasets(save_path)
    except FileNotFoundError:
        print(f'File not found: {file}')
        continue


from DataProcessing.TradeSplitter import TradeSplitter
from multiprocessing import cpu_count

trades_dir = 'data/raw/trades'
save_dir = 'data/processed/split_trades'
stock_info_path = 'data/raw/stocks.csv'

n_cores = cpu_count() - 1
splitter = TradeSplitter(trades_dir, save_dir, stock_info_path, n_cores=n_cores)
splitter.split_all()
from DataProcessing.TradeSplitter import TradeSplitter
from multiprocessing import cpu_count

trades_dir = 'data/zipped_trade_data'
save_dir = 'data/split_trades/'
stock_info_path = 'data/stock_info.csv'
exchange_times_path = 'data/exchange_trading_times.csv'

n_cores = cpu_count()
splitter = TradeSplitter(stock_info_path, exchange_times_path, save_dir, n_cores)
splitter.process_all(trades_dir)
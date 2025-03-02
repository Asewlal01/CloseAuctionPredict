from src.DataProcessing.TradeSplitter import TradeSplitter

trades_dir = '../data/raw/trades'
save_dir = '../data/processed/split_trades'
stock_info_path = '../data/raw/stocks.csv'

splitter = TradeSplitter(trades_dir, save_dir, stock_info_path, n_cores=4, core_files=4)
splitter.split_all()
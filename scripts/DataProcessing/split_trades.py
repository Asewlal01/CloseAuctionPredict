from DataProcessing.TradeSplitter import TradeSplitter

trades_dir = 'data/raw/trades'
save_dir = 'data/processed/split_trades'
stock_info_path = 'data/raw/stocks.csv'

splitter = TradeSplitter(trades_dir, save_dir, stock_info_path)
splitter.split_all()
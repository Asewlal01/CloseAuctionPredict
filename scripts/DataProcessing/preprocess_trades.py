from DataProcessing.TradePreprocessor import TradePreprocessor

save_dir = 'data/aggregated_trades'
stock_info_path = 'data/stock_info.csv'
exchange_times_path = 'data/exchange_trading_times.csv'
interval = '1min'
n_cores = 8
trade_processor = TradePreprocessor(save_dir, stock_info_path, exchange_times_path, interval, n_cores)

trade_path = 'data/zipped_trade_data'
trade_processor.process_all(trade_path)


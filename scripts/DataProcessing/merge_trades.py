from DataProcessing.TradeMerger import TradeMerger

sequence_size = 7 * 60
trades_dir = 'data/processed/aggregated_trades'
save_dir = 'data/processed/merged_trades'
trade_dates = 'data/processed/dates.csv'
train_months = 0.75
val_months = 0.15

merger = TradeMerger(sequence_size, trades_dir, save_dir, trade_dates, train_months, val_months)
merger.add_socks()
merger.split_data()
merger.save_data()
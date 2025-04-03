from DataProcessing.TradeMerger import TradeMerger

trades_dir = 'data/aggregated_trades'
save_dir = 'data/merged_trades'

trade_merger = TradeMerger(trades_dir, save_dir)
trade_merger.process_all()


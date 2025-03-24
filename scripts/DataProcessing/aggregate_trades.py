from DataProcessing.TradeAggregator2 import TradeAggregator

trade_dir = 'data/split_trades'
save_dir = 'data/aggregated_trades'
aggregation_interval = '1min'

aggregator = TradeAggregator(save_dir, aggregation_interval)
aggregator.aggregate_all(trade_dir)




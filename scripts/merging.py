import os
from src.AggregationMerger import AggregationMerger

# Path to the directory containing the aggregated trade data
aggregated_trades_path = '../data/processed/aggregated_trades'
save_path = '../data/processed/merged_trades'
stocks = os.listdir(aggregated_trades_path)

# Merger object
n_steps = 442
train_split = 0.8
validation_split = 0.1
merger = AggregationMerger(n_steps, train_split, validation_split)

for i, stock in enumerate(stocks):
    stock_path = os.path.join(aggregated_trades_path, stock)
    merger.add_stock(stock, stock_path)

    print(f"Added stock {stock} - {(i+1) / len(stocks) * 100:.2f}%")

# Saving
merger.save_data(save_path)


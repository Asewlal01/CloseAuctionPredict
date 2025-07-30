from DataProcessing.FeatureEngineers.ClosingFeatureEngineer import ClosingFeatureEngineer


# Paths for processed data
lob_path = 'example_data/processed/quote_data'
trade_path = 'example_data/processed/trade_data'
auction_path = 'example_data/processed/auction_data'

# Path to save the engineered data
save_path = 'example_data/engineered/'

# What dates are we interested in?
start_date = [2023, 1]
end_date = [2023, 2]

# How long is the sequence we want to consider in minutes?
sequence_size = 60

# How long is the prediction horizon in minutes?
horizon = 5

# Perform the feature engineering
engineer = ClosingFeatureEngineer(lob_path, trade_path, auction_path, save_path, sequence_size, horizon)
engineer.assemble_dataset(start_date, end_date)


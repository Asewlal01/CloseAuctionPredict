from DataProcessing.FeatureEngineers.IntradayFeatureEngineer import IntradayFeatureEngineer


# Add the (absolute) path to the processed data directory of the limit order book
path_to_data = 'data/processed/limit_order_book/'

# Add the (absolute) path to the directory where the feature engineered data will be saved
path_to_save = 'data/engineered/intraday/'

# The parameters for the feature engineer
sequence_size = 420     # The number of time steps to include in each sequence
horizon = 5             # The number of time steps to predict in the future
samples_to_keep = 5     # The number of samples to keep for each stock on each day
start_date = [2022, 1]  # The start date for the feature engineering (year, month)
end_date = [2022, 12]   # The end date for the feature engineering (year, month)

# Create the feature engineer
feature_engineer = IntradayFeatureEngineer(path_to_data, path_to_save,
                                           sequence_size, horizon, samples_to_keep)

# Run the feature engineering
feature_engineer.assemble(start_date, end_date)




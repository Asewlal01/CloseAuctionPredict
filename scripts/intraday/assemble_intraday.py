from DataProcessing.DatasetAssemblers.IntradayAssembler import IntradayAssembler

# Add the (absolute) path to the processed data directory of the limit order book
path_to_data = 'data/engineered/intraday/'

# Add the (absolute) path to the directory where the given dataset will be saved
save_path = 'data/datasets/intraday/'

# Parameters
train_size = 3  # Number of months to use for training
assembler = IntradayAssembler(path_to_data, save_path, train_size)

start_date = [2022, 1] # The start date for the dataset (year, month)
end_date = [2022, 4] # The end date for the dataset (year, month)
assembler.assemble(start_date, end_date)

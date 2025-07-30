from DataProcessing.DatasetAssemblers.ClosingAssembler import ClosingAssembler

# Path to the feature engineered data set
path_to_data = 'example_data/engineered'

# Path to the directory where the assembled dataset will be saved
save_base_path = 'example_data/dataset/'


assembler = ClosingAssembler(path_to_data, save_base_path, prediction_horizon=0)
assembler.assemble(days_to_normalize=5)
from DataProcessing.DatasetAssembler import DatasetAssembler

trades_dir = '/home/amish/Projects/CloseAuctionPredict/data/aggregated_files'
save_dir = '/home/amish/Projects/CloseAuctionPredict/data/merged_files'

start_date = [2021, 11]
end_date = [2022, 12]

dataset_assembler =  DatasetAssembler(trades_dir, save_dir, sequence_size=420)
dataset_assembler.process_all(start_date, end_date)


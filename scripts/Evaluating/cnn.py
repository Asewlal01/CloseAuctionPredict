from Models.ConvolutionalModels.LimitOrderBookConvolve import LimitOrderBookConvolve
from Modeling.DatasetManager import DatasetManager
from Modeling.WalkForwardTester import WalkForwardTester
import os
import pandas as pd
from tqdm import tqdm

def main(model, name):
    path_to_data = 'data/merged_files'
    results = f'results/evaluation/{name}'
    os.makedirs(results, exist_ok=True)

    # Setup dataset manager
    dataset = DatasetManager(path_to_data, 12)
    dataset.setup_dataset('2021-12')

    # Setup tester
    tester = WalkForwardTester(model, dataset)

    # Training parameters
    lr = 1e-4
    epochs = 20
    verbose = False

    train_results = []
    test_results = []
    test_months = []

    for _ in tqdm(range(12)):
        tester.train(epochs, lr, verbose)
        train_evaluation = tester.evaluate_on_train()
        test_evaluation = tester.evaluate_on_test()

        # Save results
        test_month = dataset.test_month
        train_results.append(train_evaluation)
        test_results.append(test_evaluation)
        test_months.append(test_month)

        # Increment and reset
        model.reset_parameters()
        try:
            dataset.increment_dataset()
        except:
            print("No more data to increment.")
            break

    # Save results to csv
    train_df = pd.DataFrame(train_results, columns=['Profit', 'Accuracy', 'Loss'], index=test_months)
    test_df = pd.DataFrame(test_results, columns=['Profit', 'Accuracy', 'Loss'], index=test_months)

    train_df_path = os.path.join(results, 'train_results.csv')
    test_df_path = os.path.join(results, 'test_results.csv')
    train_df.to_csv(train_df_path, index=True, index_label='Month')
    test_df.to_csv(test_df_path, index=True, index_label='Month')
    print("Evaluation completed for CNN.")

if __name__ == '__main__':
    model = LimitOrderBookConvolve(
        sequence_size=420,
        conv_channels=[64, 32, 16],
        fc_neurons=[32, 16],
        kernel_size=[3, 5, 7],
        dropout=0.1
    )
    model.to('cuda')

    model_name = 'CNN'
    main(model, model_name)









import os
import pandas as pd


class AggregationMerger:
    def __init__(self, n_steps, train_split, validation_split):
        """
        Constructor for AggregationMerger class.

        Parameters
        ----------
        n_steps : Number of time steps to take before the closing auction
        train_split : Proportion of days to use for training
        validation_split : Proportion of days to use for validation
        """
        self.n_steps = n_steps
        self.train_split = train_split
        self.validation_split = validation_split

        self.train_set = {
            'price': [],
            'quantity': [],
            'indices': [],
        }
        self.validation_set = {
            'price': [],
            'quantity': [],
            'indices': [],
        }
        self.test_set = {
            'price': [],
            'quantity': [],
            'indices': [],
        }

    def add_stock(self, stock, stock_path):
        """
        Add trading data from the specified stock to the training, validation and test sets. This method does not check if
        the size of the dataframes is greater or equal to the number of steps specified in the constructor to maximize the performance

        Parameters
        ----------
        stock : Name of the stock
        stock_path : Path to the stock directory
        """

        files = os.listdir(stock_path)
        file_split = self.split_files(files)
        for name, group in file_split.items():
            for file in group:
                file_path = os.path.join(stock_path, file)
                df = self.load_data(file_path)

                # Print if it has atleast one missing value
                if df['price'].isnull().values.any():
                    # If the volume also has missing values, then something is wrong
                    if df['quantity'].isnull().values.any():
                        print(f"Missing values in {file_path}")
                    # Ffill with the previous value
                    else:
                        df['price'].ffill()

                # Adding the data to the corresponding sets
                self.add_data(df, name)

        # Add the lengths to indices
        self.train_set['indices'].append([stock, len(self.train_set['price'])])
        self.validation_set['indices'].append([stock, len(self.validation_set['price'])])
        self.test_set['indices'].append([stock, len(self.test_set['price'])])

    def load_data(self, file_path):
        """
        Load data from the specified file path. This method also performs a cutoff to ensure that the data has the same size as the number of steps

        Parameters
        ----------
        file_path : Path to the file to load

        Returns
        -------
        """
        df = pd.read_parquet(file_path)
        df = df.iloc[-self.n_steps:]

        return df

    def split_files(self, files):
        """
        Split the files into three sets: training, validation and test

        Parameters
        ----------
        files : List of files to split

        Returns
        -------
        Training, validation and test files
        """
        n_files = len(files)
        n_train = int(n_files * self.train_split)
        n_validation = int(n_files * self.validation_split)

        train_files = files[:n_train]
        validation_files = files[n_train:n_train + n_validation]
        test_files = files[n_train + n_validation:]

        return {
            'train': train_files,
            'validation': validation_files,
            'test': test_files
        }

    def add_data(self, df, group):
        """
        Add the training, validation and test data to the corresponding sets

        Parameters
        ----------
        df : Dataframe to add to the sets
        group : Group to add the data to (train, validation, test)
        """

        price_data = df['price'].values
        quantity_data = df['quantity'].values

        if group == 'train':
            self.train_set['price'].append(price_data)
            self.train_set['quantity'].append(quantity_data)
        elif group == 'validation':
            self.validation_set['price'].append(price_data)
            self.validation_set['quantity'].append(quantity_data)
        elif group == 'test':
            self.test_set['price'].append(price_data)
            self.test_set['quantity'].append(quantity_data)
        else:
            raise ValueError(f"Invalid group: {group}")

    def return_data_df(self):
        """
        Return the training, validation and test sets as dataframes

        Returns
        -------
        Tuple containing the training, validation and test sets
        """
        train_df = {}
        validation_df = {}
        test_df = {}

        for key in self.train_set.keys():
            train_df[key] = pd.DataFrame(self.train_set[key])
            validation_df[key] = pd.DataFrame(self.validation_set[key])
            test_df[key] = pd.DataFrame(self.test_set[key])

        return train_df, validation_df, test_df

    def save_data(self, save_path):
        """
        Save the training, validation and test sets to the specified path

        Parameters
        ----------
        save_path : Path to save the data to
        """

        # Creating folders to save the data
        train_path = os.path.join(save_path, 'train')
        validation_path = os.path.join(save_path, 'validation')
        test_path = os.path.join(save_path, 'test')

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        train_df, validation_df, test_df = self.return_data_df()

        for key in train_df.keys():
            train_df[key].to_parquet(os.path.join(train_path, f"{key}.parquet"))
            validation_df[key].to_parquet(os.path.join(validation_path, f"{key}.parquet"))
            test_df[key].to_parquet(os.path.join(test_path, f"{key}.parquet"))

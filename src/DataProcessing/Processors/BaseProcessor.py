import pandas as pd
import os

class BaseProcessor:
    """
    This abstract class defines the interface for file processing. File processors should inherit from this class
    and implement the process_file method. The file to be loaded is assumed to be a gzip compressed CSV file.
    """
    def __init__(self, file_path: str):
        """
        Initialize the FileProcessor with the path to the file to be processed.

        Parameters
        ----------
        file_path : Path to the file to be processed.
        """
        self.file_path: str = file_path

        self.df: pd.DataFrame = pd.read_csv(self.file_path, compression='gzip')
        self.convert_to_time()

        self.aggregated_data: dict[str, pd.DataFrame] = {}

    def convert_to_time(self) -> None:
        """
        Convert the 't' column to datetime format.
        """
        self.df['t'] = pd.to_datetime(self.df['t'], format='mixed')

    def process_file(self):
        """
        Process the file given dataframe. This class should be implemented by subclasses to define specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement process_file method.")

    def save_results(self, save_path):
        """
        Save the processed results to a file. This method should be implemented by subclasses to define specific
        saving logic.
        """
        # Check if data has been aggregated
        if self.aggregated_data is None or len(self.aggregated_data) == 0:
            print("Aggregated data is empty. Cannot save results. Call process_file() first.")
            return

        os.makedirs(save_path, exist_ok=True)
        for date, df in self.aggregated_data.items():
            # Create the file name
            file_name = f"{date}.parquet"
            file_path = os.path.join(save_path, file_name)

            # Save the dataframe to a parquet file
            df.to_parquet(file_path)

    def get_id(self) -> str:
        """
        Get the id of the file as this is used to identify the file in the dataset.

        Returns
        -------
        ID of the file as a string.
        """
        # Get the file name
        file_path = self.file_path

        # The id is placed after the last underscore in the file name
        idx = file_path.split('_')[-1]

        # idx still contains the file extension
        idx = idx.split('.')[0]

        return idx



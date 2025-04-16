import pandas as pd

class FileProcessor:
    """
    This abstract class defines the interface for file processing. File processors should inherit from this class
    and implement the process_file method. The file to be loaded is assumed to be a gzip compressed CSV file.
    """
    def __init__(self, file_path: str,):
        """
        Initialize the FileProcessor with the path to the file to be processed.

        Parameters
        ----------
        file_path : Path to the file to be processed.
        """
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path, compression='gzip')
        self.convert_to_time()
        self.process_file()

    def convert_to_time(self) -> None:
        """
        Convert the 't' column to datetime format.
        """
        self.df['t'] = pd.to_datetime(self.df['t'], format='mixed')

    def process_file(self):
        """
        Process  the file. This method should be implemented by subclasses to define specific processing logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")
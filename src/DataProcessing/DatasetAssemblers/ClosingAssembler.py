from DataProcessing.DatasetAssemblers.BaseAssembler import BaseAssembler

VARIABLES = ['z']  # Variables to load from the dataset

class ClosingAssembler(BaseAssembler):
    """
    Assembler for closing datasets.
    """
    def __init__(self, dataset_path: str, save_path: str, prediction_horizon: int=1) -> None:
        """
        Initialize the ClosingAssembler with input and output paths.

        Parameters
        ----------
        dataset_path : Path to the dataset directory containing the files.
        save_path : Path to save the assembled dataset. If None, defaults to dataset_path (not recommended) due to
        potential overwriting of original data.
        prediction_horizon : The number of time steps to predict into the future, by default 1.
        """
        super().__init__(dataset_path, save_path, VARIABLES, prediction_horizon)
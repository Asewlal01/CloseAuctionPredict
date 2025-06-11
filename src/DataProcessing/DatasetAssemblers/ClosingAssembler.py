import torch
import os
from DataProcessing.DatasetAssemblers.IntradayAssembler import compute_statistics as compute_time_statistics
from DataProcessing.DatasetAssemblers.IntradayAssembler import normalize_dataset as normalize_time_dataset
from tqdm import tqdm
import shutil

class ClosingAssembler:
    """
    Assembler for closing datasets.
    """

    def __init__(self, dataset_path: str, save_path: str = None):
        """
        Initialize the ClosingAssembler with the dataset path and save path.

        Parameters
        ----------
        dataset_path : Path to the dataset directory containing the files.
        save_path : Path to save the assembled dataset. If None, defaults to dataset_path (not recommended) due to
        potential overwriting of original data.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.files = sorted(os.listdir(dataset_path))

        self.save_path = save_path if save_path else dataset_path
        os.makedirs(self.save_path, exist_ok=True)

    def assemble(self, start_date: str, train_size: int, validation_size: int= 0, test_size: int=0):
        """
        Assemble the dataset for the given date range.

        Parameters
        ----------
        start_date : Starting date for the dataset, in the format year-month-day (e.g., '2022-01-01').
        train_size : Size of the training dataset.
        validation_size : Size of the validation set, in number of samples.
        test_size : Size of the test set, in number of samples.

        Returns
        -------
        None
        """
        # Get the files within the given date
        total_size = train_size + validation_size + test_size
        dates = self._get_files(start_date, total_size)

        # Split into train, validation, and test sets
        train_dates = dates[:train_size]
        x_concatenated, z_concatenated = load_input_concatenated(train_dates, self.dataset_path)

        # Compute z-statistics for normalization
        x_mean, x_std = compute_time_statistics(x_concatenated)
        z_mean, z_std = compute_exogenous_statistics(z_concatenated)
        del x_concatenated, z_concatenated  # Clear memory to avoid memory errors or memory overflow

        # Normalize each date's data
        for date in tqdm(dates, desc="Processing dates"):
            process_date(date, self.dataset_path, self.save_path,
                         x_mean, x_std, z_mean, z_std)

    def _get_files(self, start_date: str, size: int) -> list:
        """
        Load the files from the dataset directory that fall within the specified date range.

        Parameters
        ----------
        start_date : Starting date for the dataset, in the format year-month-day (e.g., 2022-01-01).
        size: Number of files to retrieve starting from the given date.

        Returns
        -------
        List of file paths that fall within the specified date range.
        """
        # Check if the start_date is in files
        if start_date not in self.files:
            raise ValueError(f"No files found for the start date {start_date} in the dataset directory.")

        # Find the index of the start date
        start_index = self.files.index(start_date)

        # Get the files starting from the start date
        if start_index + size > len(self.files):
            raise ValueError(f"Requested size {size} exceeds available files from start date {start_date}.")

        return self.files[start_index:start_index + size]

def load_input_concatenated(dates: list[str], dataset_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load all the input tensors over all the specified dates and concatenate them into a single tensor.

    Parameters
    ----------
    dates : List of date strings in the format 'YYYY-MM-DD'.
    dataset_path : Path to the directory containing the dataset files.

    Returns
    -------
    Concatenated X and z tensors for the given set.
    """
    xs = []
    zs = []

    for date in dates:
        file_path = os.path.join(dataset_path, f"{date}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data for date {date} not found at {file_path}")

        # Load x and z
        x = torch.load(os.path.join(file_path, 'x.pt'), weights_only=False)
        z = torch.load(os.path.join(file_path, 'z.pt'), weights_only=False)
        xs.append(x)
        zs.append(z)

    # Concatenation
    x_concatenated = torch.cat(xs, dim=0)
    z_concatenated = torch.cat(zs, dim=0)

    return x_concatenated, z_concatenated

def compute_exogenous_statistics(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation of the exogenous variables.

    Parameters
    ----------
    z : Tensor containing the exogenous variables.

    Returns
    -------
    Tuple containing the mean and standard deviation tensors.
    """

    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True)

    return mean, std

def normalize_exogenous_dataset(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize the exogenous dataset using the provided mean and standard deviation.

    Parameters
    ----------
    z : Tensor containing the exogenous variables.
    mean : Mean tensor for normalization.
    std : Standard deviation tensor for normalization.

    Returns
    -------
    Normalized tensor of exogenous variables.
    """
    if std is None or std.sum() == 0:
        raise ValueError("Standard deviation cannot be zero for normalization.")

    return (z - mean) / std

def process_date(date: str, dataset_path: str, save_path: str,
                 x_mean: torch.Tensor, x_std: torch.Tensor, z_mean: torch.Tensor, z_std: torch.Tensor):
    """
    Process a single date of data, normalizing the input tensors and saving them to the specified path.

    Parameters
    ----------
    date : Date to process in the format 'YYYY-MM-DD'.
    dataset_path : Path to the directory containing the dataset for the specified date.
    save_path : Path to save the processed dataset.
    x_mean : Mean tensor for normalizing x.
    x_std : Standard deviation tensor for normalizing x.
    z_mean : Mean tensor for normalizing z.
    z_std : Standard deviation tensor for normalizing z.
    """

    # Path to the given date
    date_path = os.path.join(dataset_path, f"{date}")
    if not os.path.exists(date_path):
        raise FileNotFoundError(f"Data for date {date} not found at {date_path}")

    # Load the input tensors
    x = torch.load(os.path.join(date_path, 'x.pt'), weights_only=False)
    z = torch.load(os.path.join(date_path, 'z.pt'), weights_only=False)

    # Normalize the datasets
    x_normalized = normalize_time_dataset(x, x_mean, x_std)
    z_normalized = normalize_exogenous_dataset(z, z_mean, z_std)

    # Skip if we have any nan
    if torch.isnan(x_normalized).any() or torch.isnan(z_normalized).any():
        raise ValueError(f"NaN values found in normalized tensors for date {date}. Please check the input data.")

    # Save the normalized tensors
    save_date_path = os.path.join(save_path, f"{date}")
    os.makedirs(save_date_path, exist_ok=True)

    torch.save(x_normalized, os.path.join(save_date_path, 'x.pt'))
    torch.save(z_normalized, os.path.join(save_date_path, 'z.pt'))

    # Copy y from the original data to same folder without loading
    y_load_path = os.path.join(date_path, 'y.pt')
    y_save_path = os.path.join(save_date_path, 'y.pt')
    if os.path.exists(y_load_path):
        shutil.copy(y_load_path, y_save_path)
    else:
        raise FileNotFoundError(f"Output tensor y not found at {y_load_path}")
import torch
import os
# from DataProcessing.FeatureEngineers.IntradayFeatureEngineer import date_type, generate_dates
import gc
from tqdm import tqdm
import shutil

date_type = tuple
generate_dates = lambda x : x

class IntradayAssembler:
    """
    Assembles intraday datasets from raw data.
    """
    def __init__(self, data_path: str, save_path: str, train_size: int=3):
        self.data_path = data_path
        self.save_path = save_path
        self.train_size = train_size

    def assemble(self, start_date: date_type, end_date: date_type):
        # Generate the months to process
        months = generate_dates(start_date, end_date)

        # First parts are used for training, second for testing
        training_months = months[:self.train_size]

        # Load the training dataset
        x_concatenated = load_input_concatenated(training_months, self.data_path)

        # Computing z-statistics for normalization
        mean, std = compute_statistics(x_concatenated)
        del x_concatenated  # Ensure memory is cleared to avoid memory errors

        for month in tqdm(months):
            # Process each month
            process_month(month, self.data_path, self.save_path, mean, std)

def load_input_concatenated(months: list[date_type], data_path: str) -> torch.Tensor:
    """
    Load all the input tensors over all the specified months and concatenate them into a single tensor.

    Parameters
    ----------
    months : Months to load the dataset for, in the format [[year, month], ...]
    data_path : Path to the directory containing the dataset for the specified months

    Returns
    -------
    Concatenated X and y tensors for the give set
    """
    # Check if the months list is empty
    if len(months) < 1:
        raise ValueError("At least one month must be specified to load the dataset.")

    # Lists to store data
    xs = []

    for month in months:
        # Path to the given months
        month_path = os.path.join(data_path, f"{month[0]}-{month[1]}")
        if not os.path.exists(month_path):
            raise FileNotFoundError(f"Data for month {month} not found at {month_path}")

        # Getting all the days within the month
        days = os.listdir(month_path)
        for day in days:
            # Day path
            day_path = os.path.join(month_path, day)

            # Loading
            x = torch.load(os.path.join(day_path, 'x.pt'), weights_only=False)

            # Adding the data to the lists
            xs.append(x)

    # Concatenate all the data
    xs = torch.cat(xs, dim=0)

    # Clear memory to ensure no memory errors occur due to large dataset
    del x
    gc.collect()

    return xs

def compute_statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the z-statistics for the given input tensor. It is assumed that the input tensor is three-dimensional,
    with (samples, time steps, features). Statistics are computed by first transposing the tensor to
    (samples*timesteps, features) and then computing the mean and standard deviation for each feature.

    Parameters
    ----------
    x : Input tensor of shape (samples, time steps, features)

    Returns
    -------
    Mean and standard deviation tensors of shape (1, 1, features) for future broadcasting.
    """

    # Reshaping to 2D tensor
    x_reshaped = x.view(-1, x.shape[-1])

    # Computing the mean and standard deviation
    mean = x_reshaped.mean(dim=0, keepdim=True)
    std = x_reshaped.std(dim=0, keepdim=True)

    # Add dimensions for broadcasting
    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0)

    return mean, std

def normalize_dataset(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize the dataset using the given mean and standard deviation.

    Parameters
    ----------
    x : Input tensor of shape (samples, time steps, features)
    mean : Mean tensor of shape (1, 1, features)
    std : Standard deviation tensor of shape (1, 1, features)

    Returns
    -------
    Normalized tensor of the same shape as input.
    """

    # Normalizing the dataset
    return (x - mean) / std

def process_month(month: date_type, data_path: str, save_path: str, mean: torch.Tensor, std: torch.Tensor):
    """
    Process a single month of data, normalizing the input tensors and saving them to the specified path.

    Parameters
    ----------
    month : Month to process in the format [year, month]
    data_path : Path to the directory containing the dataset for the specified month
    save_path : Path to save the processed dataset
    mean : Mean tensor for normalization
    std : Standard deviation tensor for normalization
    """

    # Path to the given month
    month_path = os.path.join(data_path, f"{month[0]}-{month[1]}")
    if not os.path.exists(month_path):
        raise FileNotFoundError(f"Data for month {month} not found at {month_path}")

    # Path to saving
    month_save_path = os.path.join(save_path, f"{month[0]}-{month[1]}")

    # Going through all the days
    for day in os.listdir(month_path):
        # Path to the day
        day_path = os.path.join(month_path, day)

        # Load the input tensor for the day
        x = torch.load(os.path.join(day_path, 'x.pt'), weights_only=False)

        # Normalize the dataset
        x_normalized = normalize_dataset(x, mean, std)

        # Skip if nan
        if torch.isnan(x_normalized).any():
            print(f"Skipping day {day} in month {month} due to NaN values in normalized data.")
            continue

        # Save the normalized tensor
        day_save_path = os.path.join(month_save_path, day)
        os.makedirs(day_save_path, exist_ok=True)

        torch.save(x_normalized, os.path.join(day_save_path, 'x.pt'))

        # Copy y from the original data to same folder without loading
        y_load_path = os.path.join(day_path, 'y.pt')
        y_save_path = os.path.join(day_save_path, 'y.pt')
        if os.path.exists(y_load_path):
            shutil.copy(y_load_path, y_save_path)
        else:
            raise FileNotFoundError(f"Output tensor y not found at {y_load_path}")
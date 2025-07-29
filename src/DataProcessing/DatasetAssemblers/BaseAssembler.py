import os
import torch
import shutil
from tqdm import tqdm

statistics_type = tuple[torch.Tensor, torch.Tensor]  # Tuple for mean and std tensors

class BaseAssembler:
    """
    Base class for dataset assemblers. This class provides a template for assembling datasets
    from raw data, including methods for normalization and processing.
    """

    def __init__(self, dataset_path: str, save_path: str = None, variables: list[str] = None,
                 prediction_horizon: int = 0):
        """
        Initialize the ClosingAssembler with the dataset path and save path.

        Parameters
        ----------
        dataset_path : Path to the dataset directory containing the files.
        save_path : Path to save the assembled dataset. If None, defaults to dataset_path (not recommended) due to
        potential overwriting of original data.
        variables : List of variable names to load from the dataset (e.g., ['z1', 'z2']). If None, defaults to an empty list.
        prediction_horizon : The prediction horizon for the dataset, default is 1. Removes the last time steps from the
        dataset equal to the prediction horizon.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.files = sorted(os.listdir(dataset_path))
        self.variables = variables if variables is not None else []
        self.prediction_horizon = prediction_horizon

        self.save_path = save_path if save_path else dataset_path
        os.makedirs(self.save_path, exist_ok=True)

    def assemble(self, days_to_normalize: int=1) -> None:
        """
        Assemble the dataset from raw data. This method should be implemented by subclasses.

        Parameters
        ----------
        days_to_normalize : Number of days to use for normalization
        """

        dates = os.listdir(self.dataset_path)
        dates = sorted(dates)  # Ensure dates are sorted

        # Normalize each date's data
        for i in tqdm(range(days_to_normalize, len(dates)), desc="Processing dates"):
            # Get current and previous dates
            date = dates[i]
            previous_dates = dates[i - days_to_normalize:i]

            process_date(date, previous_dates, self.variables, self.dataset_path, self.save_path, self.prediction_horizon)

def load_input_concatenated(dates: list[str], dataset_path: str, exogenous_variables: list[str],
                            prediction_horizon: int) -> tuple[torch.Tensor, ...]:
    """
    Load all the input tensors over all the specified dates and concatenate them into a single tensor.

    Parameters
    ----------
    dates : List of date strings in the format 'YYYY-MM-DD'.
    dataset_path : Path to the directory containing the dataset files.
    exogenous_variables: The list of exogenous variables to load (e.g., ['z1', 'z2']).
    prediction_horizon : The prediction horizon for the dataset, default is 1.

    Returns
    -------
    Concatenated X and z tensors for the given set.
    """

    xs = []
    exogenous_variables = {exogenous_variable: [] for exogenous_variable in exogenous_variables}
    for date in dates:
        file_path = os.path.join(dataset_path, f"{date}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data for date {date} not found at {file_path}")

        # Loading x
        x_loaded = load_variable(file_path, 'x')
        if prediction_horizon > 0:
            x_loaded = x_loaded[:, :-prediction_horizon, :]  # Remove the last time steps equal to the prediction horizon
        xs.append(x_loaded)

        # Loading each variable
        for variable in exogenous_variables.keys():
            exogenous_loaded = load_variable(file_path, variable)
            exogenous_variables[variable].append(exogenous_loaded)

    # Concatenate all tensors
    xs = torch.cat(xs, dim=0)
    variable_tensors = {variable: torch.cat(tensors, dim=0) for variable, tensors in exogenous_variables.items()}

    return xs, *tuple(variable_tensors.values())

def load_variable(path: str, variable: str) -> torch.Tensor:
    """
    Load a specific variable from the dataset directory.

    Parameters
    ----------
    path : Path to the directory containing the dataset files.
    variable : Name of the variable to load (e.g., 'x', 'z1', 'z2').

    Returns
    -------
    Loaded tensor for the specified variable.
    """
    variable_path = os.path.join(path, f'{variable}.pt')
    if not os.path.exists(variable_path):
        raise FileNotFoundError(f"Variable {variable} not found at {variable_path}")

    return torch.load(variable_path, weights_only=False)

def save_variable(path: str, variable: str, tensor: torch.Tensor) -> None:
    """
    Save a specific variable tensor to the dataset directory.

    Parameters
    ----------
    path : Path to the directory where the dataset files will be saved.
    variable : Name of the variable to save (e.g., 'x', 'z1', 'z2').
    tensor : Tensor to save for the specified variable.
    """

    os.makedirs(path, exist_ok=True)
    variable_path = os.path.join(path, f'{variable}.pt')
    torch.save(tensor, variable_path)

def compute_time_statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation of the time variable.

    Parameters
    ----------
    x : Tensor containing the time variable.

    Returns
    -------
    Tuple containing the mean and standard deviation tensors.
    """
    if x.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    mean = x.mean(dim=(0,1), keepdim=True)
    std = x.std(dim=(0,1), keepdim=True)

    return mean, std

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

def normalize_time_variable(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize the time variable using the given mean and standard deviation.

    Parameters
    ----------
    x : Input tensor of shape (samples, time steps, features)
    mean : Mean tensor of shape (1, features)
    std : Standard deviation tensor of shape (1, features)

    Returns
    -------
    Normalized tensor of the same shape as input.
    """
    if std is None or std.any() == 0:
        raise ValueError("Standard deviation cannot be zero for normalization.")

    return (x - mean) / std

def normalize_exogenous_variable(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize the exogenous variable using the provided mean and standard deviation.

    Parameters
    ----------
    z : Tensor containing the exogenous variables.
    mean : Mean tensor for normalization.
    std : Standard deviation tensor for normalization.

    Returns
    -------
    Normalized tensor of exogenous variables.
    """
    if std is None or std.any() == 0:
        raise ValueError("Standard deviation cannot be zero for normalization.")

    return (z - mean) / std

def process_date(date: str, previous_dates: list[str],
                 variables: list[str], dataset_path: str, save_path: str, prediction_horizon: int) -> None:
    """
    Process a single date of data, normalizing the input tensors and saving them to the specified path.

    Parameters
    ----------
    date : Date to process in the format 'YYYY-MM-DD'.
    previous_dates : List of previous dates to use for normalization.
    variables : List of variable names to load from the dataset (e.g., ['z1', 'z2']).
    dataset_path : Path to the directory containing the dataset for the specified date.
    save_path : Path to save the processed dataset.
    prediction_horizon : The prediction horizon for the dataset, default is 1.
    """
    # Path to the given date
    date_path = os.path.join(dataset_path, f"{date}")
    if not os.path.exists(date_path):
        raise FileNotFoundError(f"Data for date {date} not found at {date_path}")
    date_save_path = os.path.join(save_path, f"{date}")

    # Statistics for normalization
    x_previous, *z_previous = load_input_concatenated(previous_dates, dataset_path, variables, prediction_horizon)
    x_statistics = compute_time_statistics(x_previous)
    z_statistics = {variable: compute_exogenous_statistics(z) for variable, z in zip(variables, z_previous)}

    # Normalization of x
    x = load_variable(date_path, 'x')
    if prediction_horizon > 0:
        x = x[:, :-prediction_horizon, :]  # Remove the last time steps equal to the prediction horizon
    x_normalized = normalize_time_variable(x, *x_statistics)
    save_variable(date_save_path, 'x', x_normalized)

    # Normalization of each exogenous variable
    for variable in variables:
        z = load_variable(date_path, variable)
        z_normalized = normalize_exogenous_variable(z, *z_statistics[variable])
        save_variable(date_save_path, variable, z_normalized)

    # We also copy y and the stock_ids to the same folder
    y_load = os.path.join(date_path, 'y.pt')
    y_save = os.path.join(date_save_path, 'y.pt')
    shutil.copy(y_load, y_save)

    stock_ids_load = os.path.join(date_path, 'stock_ids.pt')
    stock_ids_save = os.path.join(date_save_path, 'stock_ids.pt')
    shutil.copy(stock_ids_load, stock_ids_save)



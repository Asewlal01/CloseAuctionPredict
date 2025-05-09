import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(month: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the data for a specific month. It is assumed that closing_returns are the variable to be predicted and that the
    current_returns, overnight_returns, previous_returns and relative_traded_volume are the features. Current_returns
    and relative_traded_volume can return the whole sequence.

    Parameters
    ----------
    month : Month for which to load the data.

    Returns
    -------
    Tuple containing the input and target variables.
    """
    # Check if the month exists
    if not os.path.exists(month):
        raise FileNotFoundError(f'Directory {month} does not exist.')

    # All features in the dataset
    trade_features = ['open', 'close', 'high', 'low', 'quantity', 'vwap']
    lob_prices = [f'ba{i}' for i in range(1, 6)] + [f'bb{i}' for i in range(1, 6)]
    lob_volumes = [f'bavol{i}' for i in range(1, 6)] + [f'bbvol{i}' for i in range(1, 6)]
    features = trade_features + lob_prices + lob_volumes

    # Initialize X
    df = pd.read_parquet(f'{month}/{features[0]}.parquet')
    X = df.values

    # Adding all other features of the month
    for feature in features[1:]:
        df = pd.read_parquet(f'{month}/{feature}.parquet')
        X = np.dstack([X, df.values])
    X = torch.tensor(X, dtype=torch.float)

    # Additional time-independent features
    overnight_returns = pd.read_parquet(f'{month}/overnight_return.parquet')
    previous_returns = pd.read_parquet(f'{month}/previous_return.parquet')
    z = np.hstack([overnight_returns, previous_returns])
    z = torch.tensor(z, dtype=torch.float)

    # Predictor variable
    y = pd.read_parquet(f'{month}/current_return.parquet').values
    y = torch.tensor(y, dtype=torch.float)

    # Checking for nans
    if torch.isnan(X).any():
        raise ValueError(f'NaN values found in X for month {month}')
    if torch.isnan(y).any():
        raise ValueError(f'NaN values found in y for month {month}')
    if torch.isnan(z).any():
        raise ValueError(f'NaN values found in z for month {month}')
    return X, y, z

load_path = 'raw_merged_files'
save_path = 'merged_files'

for month in tqdm(os.listdir(load_path)):
    month_load_path = os.path.join(load_path, month)

    month_save_path = os.path.join(save_path, month)
    os.makedirs(month_save_path, exist_ok=True)

    for day in os.listdir(month_load_path):
        day_load_path = os.path.join(month_load_path, day)
        X, y, z = load_data(day_load_path)

        day_save_path = os.path.join(month_save_path, day)
        os.makedirs(day_save_path, exist_ok=True)
        torch.save(X, os.path.join(day_save_path, 'X.pt'))
        torch.save(y, os.path.join(day_save_path, 'y.pt'))
        torch.save(z, os.path.join(day_save_path, 'z.pt'))

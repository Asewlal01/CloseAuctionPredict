from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager
import torch

class LimitOrderBookDatasetManager(BaseDatasetManager):
    def normalize(self) -> torch.tensor:
        """
        Normalize the dataset.

        """
        # We do not want to override the original dataset
        x, y = self.dataset
        x = x.clone()
        y = y.clone()

        # Normalize the prices
        x = normalize_prices(x)

        # Normalize the volumes
        x = normalize_volumes(x)

        # Filter the results
        x, y = filter_outliers(x, y)

        return x, y


def normalize_prices(x):
    # Price columns are even columns
    price_columns = torch.arange(0, 20, 2)

    # Min-Max is based on highest ask and lowest bid
    lowest_bid_price = x[:, :, price_columns[4]]
    highest_ask_price = x[:, :, price_columns[9]]

    # Find the max and min values
    min_price = lowest_bid_price.min(dim=1, keepdim=True).values
    max_price = highest_ask_price.max(dim=1, keepdim=True).values

    # Both tensors need to be 3D for broadcasting
    min_price = min_price.unsqueeze(1)
    max_price = max_price.unsqueeze(1)

    # Normalize the prices
    x[:, :, price_columns] = (x[:, :, price_columns] - min_price) / (max_price - min_price)

    return x

def normalize_volumes(x):
    # Volume columns are uneven columns
    volume_columns = torch.arange(1, 20, 2)

    # Min-Max is based on all the columns
    min_volume = x[:, :, volume_columns].amin(dim=(1, 2), keepdim=True)
    max_volume = x[:, :, volume_columns].amax(dim=(1, 2), keepdim=True)

    # Normalize
    x[:, :, volume_columns] = (x[:, :, volume_columns] - min_volume) / (max_volume - min_volume)

    return x

def filter_outliers(x, y):
    # Some results will have values that are greater than 1 or less than 0, which should be impossible
    # Find where x is greater than 1 or less than 0
    mask = ((x >= 0) & (x <= 1))

    # We want to get it per sample
    mask = mask.all(dim=(1, 2))

    return x[mask], y[mask]


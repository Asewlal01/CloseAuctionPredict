import torch
from Modeling.Dataset import Dataset
from Models.BaseModel import BaseModel
from Metrics.ExpectedProfit import ExpectedProfit
from Loss.GMADLLoss import GMADLLoss

class Trainer:
    """
    Class used to train models.
    """
    def __init__(self, dataset: Dataset):
        """
        Initialize the Trainer object.

        Parameters
        ----------
        dataset : Dataset object containing the data.
        """
        self.dataset = dataset

    def train(self, model: BaseModel, horizon: int, a: float, b: float,
              epochs: int=100, batch_size: int=32, patience: int=10, lr: float=1e-3,
              verbose: bool=False, device='cpu') -> dict:
        """
        Train the model. Uses early stopping to make sure that no overfitting occurs.

        Parameters
        ----------
        model : Model to train.
        horizon : Number of time steps to predict.
        a : Slope of GMADL around 0
        b : Parameter that increases the reward of higher returns
        epochs : Number of epochs to train the model.
        batch_size : Size of the batches used in training.
        device : Device to train the model on.
        patience : Number of epochs to wait before early stopping.
        lr : Learning rate to use for training.
        verbose : Boolean indicating whether to print training information.

        Returns
        -------
        Optimal model.
        """

        X, y = self.dataset.get_train_data(horizon)
        X_train, y_train = move_to_device(X, y, device)
        train_loader = create_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

        X, y = self.dataset.get_validation_data(horizon)
        X_val, y_val = move_to_device(X, y, device)

        criterion = GMADLLoss(a, b)
        validation_metric = ExpectedProfit()
        validation_metric.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_evaluation = -torch.inf
        counter = 0
        best_model = None
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item() * len(X_batch)

            total_loss /= len(train_loader)

            model.eval()
            y_pred = model(X_val).detach()
            validation_metric.optimize_tau(y_pred, y_val)
            evaluation = validation_metric(y_pred, y_val)

            if verbose:
                print(f"Epoch: {epoch + 1}: Loss: {total_loss}, Validation Metric: {evaluation}")

            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_model = model.state_dict().copy()
                counter = 0
                continue

            counter += 1
            if counter >= patience:
                break

        return best_model

def create_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int=128, shuffle=False) -> torch.utils.data.DataLoader:
    """

    Parameters
    ----------
    X : Input variables.
    y : Target variables.
    batch_size: Batch size.
    shuffle : Boolean indicating whether to shuffle the dataset.

    Returns
    -------
    Dataloader object of the given dataset.
    """

    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def move_to_device(X: torch.Tensor, y: torch.Tensor, device: str='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Moves the input and target tensors to the given device.

    Parameters
    ----------
    X : Input tensor.
    y : Target tensor.
    device : Device to move the tensors to.

    Returns
    -------
    Tuple containing the input and target tensors.
    """
    return X.to(device), y.to(device)
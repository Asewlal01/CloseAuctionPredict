from typing import Any

import torch
from Modeling.Dataset import Dataset
from Models.BaseModel import BaseModel


class Trainer:
    """
    This class handles the training of the models. It uses early stopping to prevent overfitting.
    """
    def __init__(self, dataset: Dataset):
        """
        Initialize the Trainer object.

        Parameters
        ----------
        dataset : Dataset object containing the data.
        """
        self.dataset = dataset

    def train(self, model: BaseModel, horizon: int, sequence_size: int, criterion: torch.nn.Module,
              optimizer: torch.optim.Optimizer,epochs: int=100, train_batch_size: int=32, val_batch_size: int=128,
              patience: int=10, verbose: bool=False) -> tuple[dict, float]:
        """
        Train the model. Uses early stopping to make sure that no overfitting occurs.

        Parameters
        ----------
        model : Model to train.
        horizon : Number of time steps to predict.
        sequence_size : Number of sequence steps in the input data
        criterion : Loss function to use in the training.
        optimizer : Optimizer to use in the training.
        epochs : Number of epochs to train the model.
        train_batch_size : Size of the batches used in training.
        val_batch_size : Size of the batches used in validation
        patience : Number of epochs to wait before early stopping.
        verbose : Boolean indicating whether to print training information.

        Returns
        -------
        Parameters of the best model
        """

        train_dataloader, validation_dataloader = get_loaders(self.dataset, horizon, sequence_size,
                                                              train_batch_size, val_batch_size)

        # Device is based on the model
        device = model.device

        best_loss = torch.inf
        counter = 0
        best_model = model.state_dict().copy()
        for epoch in range(epochs):
            train_loss = train_on_loader(model, train_dataloader, criterion, optimizer, device)
            val_loss = validate_on_loader(model, validation_dataloader, criterion, device)

            if verbose:
                print(f"Epoch: {epoch + 1}: Loss: {train_loss}, Validation Loss: {val_loss}")

            stop, counter, best_loss, best_model = early_stopping_check(val_loss, best_loss, patience,
                                                                        counter, model, best_model)
            if stop:
                break

        return best_model, best_loss

def create_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int=128, shuffle=False) -> torch.utils.data.DataLoader:
    """
    Create a dataloader object from the given input and target tensors.

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

def get_loaders(dataset, horizon, sequence_size, train_batch_size, val_batch_size):
    """
    Get the dataloaders for the training and validation data.

    Parameters
    ----------
    dataset : Dataset object containing the data.
    horizon : Number of time steps to predict.
    sequence_size : Number of sequence steps in the input data
    train_batch_size : Size of the batches used in training.
    val_batch_size : Size of the batches used in validation

    Returns
    -------
    Tuple containing the training and validation dataloaders.
    """
    X_train, y_train = dataset.get_train_data(horizon, sequence_size)
    train_dataloader = create_loader(X_train, y_train, batch_size=train_batch_size, shuffle=False)

    X_val, y_val = dataset.get_validation_data(horizon, sequence_size)
    validation_dataloader = create_loader(X_val, y_val, batch_size=val_batch_size, shuffle=False)

    return train_dataloader, validation_dataloader

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

def train_batch(model: BaseModel, X_batch: torch.Tensor, y_batch: torch.Tensor, criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer) -> float:
    """
    Train the model using a single batch.

    Parameters
    ----------
    model : Model to train.
    X_batch : Input tensor.
    y_batch : Target tensor.
    criterion : Loss function to use in the training.
    optimizer : Optimizer to use in the training.

    Returns
    -------
    Loss of the model.
    """
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_on_loader(model: BaseModel, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, device: str) -> float:
    """
    Train the model using a dataloader object.

    Parameters
    ----------
    model : Model to train.
    loader : Dataloader object containing the data.
    criterion : Loss function to use in the training.
    optimizer : Optimizer to use in the training.
    device : Device to use in the training.

    Returns
    -------
    Loss of the model.
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = move_to_device(X_batch, y_batch, device)
        loss = train_batch(model, X_batch, y_batch, criterion, optimizer)
        total_loss += loss * len(X_batch)
    return total_loss / len(loader)

def validate_batch(model: BaseModel, X_batch: torch.Tensor, y_batch: torch.Tensor, criterion: torch.nn.Module) -> float:
    """
    Validate the model using a single batch.

    Parameters
    ----------
    model : Model to validate.
    X_batch : Input tensor.
    y_batch : Target tensor.
    criterion : Loss function to use in the validation.

    Returns
    -------
    Loss of the model.
    """
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    return loss.item()

def validate_on_loader(model: BaseModel, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str) -> float:
    """
    Validate the model using a dataloader object.

    Parameters
    ----------
    model : Model to validate.
    loader : Dataloader object containing the data.
    criterion : Loss function to use in the validation.
    device : Device to use in the validation.

    Returns
    -------
    Loss of the model.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = move_to_device(X_batch, y_batch, device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader)

def early_stopping_check(val_loss: float, best_loss: float, patience: int, counter: int, model: BaseModel,
                         best_model: dict[str, Any] | dict) -> tuple[bool, int, float, dict[str, Any] | dict]:
    """
    Check if early stopping should be applied. It is assumed that lower validation loss is better.

    Parameters
    ----------
    val_loss : Current validation loss.
    best_loss : Best validation loss so far.
    patience : Number of epochs to wait before early stopping.
    counter : Number of epochs since the last improvement.
    model : Model to check for early stopping.
    best_model : Best model so far.

    Returns
    -------
    Tuple containing a boolean indicating whether to stop the training, the new counter, the best loss and the best model.
    """

    # We found better model
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model.state_dict().copy()
        counter = 0
    else:
        counter += 1

    # Check if we should stop
    stop = counter >= patience

    return stop, counter, best_loss, best_model
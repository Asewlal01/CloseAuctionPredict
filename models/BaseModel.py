import torch.nn as nn
import torch
from torch.utils import data
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract class that represents a model. It should be inherited by all the models in the project as it provides
    training functionality and other useful methods that are common to all the models.
    """

    def __init__(self):
        """
        Initializes the BaseModel class.
        """
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model. This method should be implemented by all the models that inherit from
        this class as it is the main method that is called when the model is used.

        Parameters
        ----------
        x : Input tensor

        Returns
        -------
        Output tensor

        """
        pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module , device: str) -> float:
        """
        Perform a training step on the model. It is assumed that x and y are from a batch of data. Does not check
        if the model is in training mode and device on which the data is located.

        Parameters
        ----------
        x : Input tensor
        y : Expected output tensor
        optimizer : Optimizer object
        criterion : Loss function
        device : Device on which the data is located

        Returns
        -------
        Loss value
        """
        # Zero the gradient before optimizing
        optimizer.zero_grad()

        # Moving to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        output = self(x)
        loss = criterion(output, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_epoch(self, train_loader: torch, optimizer, criterion, device) -> float:
        """
        Train the model for one epoch

        Parameters
        ----------
        train_loader : Training data loader
        optimizer : Optimizer object
        criterion : Loss function
        device : Device on which the data is located

        Returns
        -------
        Average loss value within the epoch

        """

        self.train()
        total_loss = 0
        for x, y in train_loader:
            batch_loss = self.train_step(x, y, optimizer, criterion, device)
            total_loss += batch_loss

        return total_loss / len(train_loader)

    def train_all(self, train_loader: data.DataLoader, validation_loader: data.DataLoader=None,
                  epochs: int=10, lr: float=0.001, verbose: bool=True, device: str='cpu') -> None:
        """
        Train multiple epochs on the model. Uses Adam as the optimizer and MSE as the loss function.

        Parameters
        ----------
        train_loader : Training data loader
        validation_loader : Validation data loader
        epochs : Number of epochs to train
        lr : Learning rate
        verbose : Print training and validation information
        device : Device on which the data is located

        Returns
        -------
        None
        """

        # Instantiate the optimizer and the loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Main loop
        for epoch in range(epochs):
            # Training loss
            train_loss = self.train_epoch(train_loader, optimizer, criterion, device)

            # Validation loss
            validation_loss = None
            if validation_loader is not None:
                validation_loss = self.evaluate(validation_loader, criterion, device)

            # Logging
            if verbose:
                print(f"Epoch {epoch + 1} - Train loss: {train_loss}")
                if validation_loss is not None:
                    print(f"Epoch {epoch + 1} - Validation loss: {validation_loss}")


    def evaluate(self, validation_loader: data.DataLoader, criterion, device: str) -> float:
        """
        Evaluate the model on the validation data

        Parameters
        ----------
        validation_loader : Validation data
        criterion : Loss function
        device: Device on which the data is located

        Returns
        -------
        Average loss value on the validation data
        """

        self.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(device)
                y = y.to(device)
                output = self(x)
                loss = criterion(output, y)
                total_loss += loss.item()

        return total_loss / len(validation_loader)



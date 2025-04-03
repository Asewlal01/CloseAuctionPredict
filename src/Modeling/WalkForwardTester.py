from typing import Any
import copy
import torch
from Models.BaseModel import BaseModel
from Modeling.DatasetManager import DatasetManager
from Metrics.ProfitCalculator import ProfitCalculator


class WalkForwardTester:
    def __init__(self, model: BaseModel, dataset_manager: DatasetManager):
        """
        Initializes the WalkForwardTester class.

        Parameters
        ----------
        model : Model to be tested
        dataset_manager : The DatasetManager instance that handles the dataset.
        """
        self.model = model
        self.dataset_manager = dataset_manager
        self.metric = ProfitCalculator()

    def train(self, epochs, batch_size, learning_rate, patience, verbose=False) -> tuple[Any, float]:
        """
        Trains the model using walk forward testing. It trains the model on the training data and validates it on the
        validation data.
        """
        # Get the training and validation data
        train_data, val_data = self.dataset_manager.get_training_data()

        # Train the model
        best_model, best_loss = train(self.model, train_data, val_data, self.metric, epochs, batch_size, learning_rate,
                                      patience, verbose)

        return best_model, best_loss

    def test(self):
        """
        Tests the model using walk forward testing. It trains the model on the training data and validates it on the
        validation data.
        """
        # Get the training and validation data
        datasets = self.dataset_manager.get_test_data()

        self.model.eval()
        with torch.no_grad():
            for dataset in datasets:
                X, y, z = dataset
                X = X.to(self.model.device)
                y = y.to(self.model.device)
                z = z.to(self.model.device)
                y_pred = self.model(X, z)
                test_metric = self.metric(y_pred, y)
                accuracy = ((y_pred * y) > 0).float().mean()

                yield test_metric.item(), accuracy.item()



def train(model, train_data, val_data, metric, epochs=10, batch_size=32, learning_rate=0.001, patience=10, verbose=True):
        """
        Trains the model on the training data and validates it on the validation data.

        Parameters
        ----------
        model : The model to be trained.
        train_data : The training data.
        val_data : The validation data.
        metric : The metric to be used for validation.
        epochs : Number of epochs to train the model.
        batch_size : Size of the batches used in training. Can be negative to use the entire dataset.
        learning_rate : Learning rate for the optimizer.
        patience : Number of epochs to wait before early stopping.
        verbose : Whether to print training information.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1)
        loss = torch.nn.BCEWithLogitsLoss()

        # Assumed to use every batch size
        if batch_size < 0:
            batch_size = len(train_data[0])

        # Converting to loader
        train_loader = create_loader(*train_data, batch_size)
        X_val, y_val, z_val = val_data

        # Instantiate the best loss and counter
        best_loss = float('inf')
        best_model = model.state_dict().copy()
        counter = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X, y, z in train_loader:
                X = X.to(model.device)
                y = y.to(model.device)
                z = z.to(model.device)
                optimizer.zero_grad()
                y_pred = model(X, z)
                loss_value = loss(y_pred, y)
                total_loss += loss_value.item() * len(y)
                loss_value.backward()
                optimizer.step()

            # Average loss
            total_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            with torch.no_grad():
                X_val = X_val.to(model.device)
                y_val = y_val.to(model.device)
                z_val = z_val.to(model.device)
                y_pred = model(X_val, z_val)
                val_metric = metric(y_pred, y_val)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss_value.item()}, Validation Loss: {val_metric.item()}")

            # Early stopping
            if -val_metric < best_loss:
                best_loss = -val_metric
                counter = 0
                best_model = copy.deepcopy(model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break

        return best_model, best_loss

def create_loader(X: torch.Tensor, y: torch.Tensor, z: torch.Tensor, batch_size: int) -> torch.utils.data.DataLoader:
    """
    Create a dataloader object from the given input and target tensors.

    Parameters
    ----------
    X : Input variables.
    y : Target variables.
    z : Additional variables.
    batch_size: Batch size.

    Returns
    -------
    Dataloader object of the given dataset.
    """

    dataset = torch.utils.data.TensorDataset(X, y, z)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from Models.BaseModel import BaseModel
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager, convert_to_classification, DatasetTuple
from Metrics.ProfitCalculator import ProfitCalculator


class WalkForwardTester:
    def  __init__(self, model: BaseModel, train_manager: BaseDatasetManager, test_manager: BaseDatasetManager,
                  sequence_size: int = 360):
        """
        Initializes the WalkForwardTester class.

        Parameters
        ----------
        model : Model to be tested
        train_manager : DatasetManager to be used for training
        test_manager : DatasetManager to be used for testing
        """
        self.model = model
        self.train_manager = train_manager
        self.test_manager = test_manager
        self.sequence_size = sequence_size
        self.metric = ProfitCalculator()

    def train(self, epochs, learning_rate, batch_size, verbose=False) -> None:
        """
        Trains the model using walk forward testing. It trains the model on the training data and validates it on the
        validation data.

        Parameters
        ----------
        epochs : Number of epochs to train the model.
        learning_rate : Learning rate to be used for training.
        batch_size : Size of the batch to be used for training the model.
        verbose : Whether to print training information.
        """
        # Get the training and validation data
        train_data = self.train_manager.get_dataset()

        # Train the model
        train(self.model, train_data, self.sequence_size, epochs, learning_rate, batch_size, verbose)

    def evaluate_on_train(self, batch_size: int=1024) -> tuple[float, float, float]:
        """
        Evaluate the model on the training data. It uses the training data to evaluate the model. The model returns
        three values: the profit, the accuracy and the loss.

        Parameters
        ----------
        batch_size: Size of the batch to be used for evaluation.

        Returns
        -------
        Profit, accuracy and loss of the model on the training data.
        """
        # Get the training and validation data
        train_data = self.train_manager.get_dataset()

        evaluation = evaluate(self.model, train_data, self.sequence_size, batch_size)

        return evaluation

    def evaluate_on_test(self, batch_size: int=1024) -> tuple[float, float, float]:
        """
        Evaluate the model on the next given month. It uses the test data to evaluate the model.

        Parameters
        ----------
        batch_size: Size of the batch to be used for evaluation.

        Returns
        -------
        Profit, accuracy and loss of the model on the test data.
        """
        # Get the training, validation and test data
        test_data = self.test_manager.get_dataset()

        evaluation = evaluate(self.model, test_data, self.sequence_size, batch_size)

        return evaluation


def train(model: BaseModel, train_data: list[DatasetTuple], sequence_size: int,
          epochs: int, learning_rate: float,
          batch_size: int = 32, verbose=True) -> None:
    """
    Train the model using the given training data. It uses the BCEWithLogitsLoss function to compute the loss and
    Adam optimizer to optimize the model. The model is trained for the given number of epochs. This training scheme
    performs multiple epoch per batch instead of multiple batch per epoch. This is done to minimize data leakage
    in the train set, which the model may exploit to overfit the data.

    Parameters
    ----------
    model : The model to be trained.
    train_data : The training data to be used for training the model.
    sequence_size : The size of the sequence to be used.
    epochs : Number of epochs to train the model.
    learning_rate : Learning rate to be used for training the model.
    batch_size : Size of the batch to be used for training the model.
    verbose : Whether to print training information.
    """

    # Initialize the loss function, optimizer and dataloaders
    loss = BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loaders = create_loader(train_data, sequence_size, batch_size)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for loader in loaders:
            for x, y, weights in loader:
                # Moving everything to the device
                x = x.to(model.device)
                y = y.to(model.device)
                weights = weights.to(model.device)

                # Training requires y to be binary
                y = convert_to_classification(y)

                # Forward pass
                y_pred = model(x)
                loss_per_sample = loss(y_pred, y)
                weighted_loss = weights * loss_per_sample
                loss_value = weighted_loss.sum()
                total_loss += loss_value.item()

                # Backward pass
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

        if verbose:
            average_loss = total_loss
            print(f"Epoch {epoch+1}/{epochs} has average Loss of : {average_loss:.4f}")

def evaluate(model: BaseModel, data: list[DatasetTuple], sequence_size: int, batch_size: int) -> tuple[float, float, float]:
    """
    Evaluate the model using the given dataset and metric. It also computes the accuracy of the model by comparing
    the sign of the predicted and actual values.

    Parameters
    ----------
    model : The model to be evaluated.
    data : The dataset to be used for evaluation.
    sequence_size : The size of the sequence to be used.
    batch_size : The size of the batch to be used for evaluation.

    Returns
    -------
    The metric value and accuracy.
    """
    # Setup
    profit_calculator = ProfitCalculator()
    loss_fn = BCEWithLogitsLoss(reduction='none')
    dataloaders = create_loader(data, sequence_size, batch_size)
    model.eval()

    with torch.no_grad():
        average_profit = 0
        average_accuracy = 0
        average_loss = 0
        total_samples = 0
        for loader in dataloaders:
            for x, y, weights in loader:
                x = x.to(model.device)
                y = y.to(model.device)
                weights = weights.to(model.device)
                y_pred = model(x)

                profit = profit_calculator(y_pred, y)

                # Accuracy checks if sign match
                accuracy = ((y_pred * y) > 0).float().sum()

                # Loss requires y to be binary
                y = convert_to_classification(y)
                loss = loss_fn(y_pred, y)
                loss = loss * weights
                loss = loss.sum()

                average_profit += profit.item() * len(y)
                average_accuracy += accuracy.item()
                average_loss += loss.item()
                total_samples += len(y)

        # Sample based
        average_profit /= total_samples
        average_accuracy /= total_samples

        return average_profit, average_accuracy, average_loss

def create_loader(dataset: list[DatasetTuple], sequence_size: int, batch_size: int) -> list[DataLoader]:
    """
    Create a DataLoader for the given dataset.

    Parameters
    ----------
    dataset : The dataset to be used for creating the DataLoader.
    sequence_size : The size of the sequence to be used
    batch_size : The batch size to be used for creating the DataLoader.

    Returns
    -------
    DataLoader for the given dataset.
    """
    dataloaders = []
    total_return = compute_total_return(dataset)
    for x, y in dataset:
        # Only keep the last sequence_size samples
        x = x[:, -sequence_size:, :]

        # Compute the sample weights
        weights = compute_sample_weights(y, total_return)

        # Convert the dataset to a PyTorch Dataset
        dataset = torch.utils.data.TensorDataset(x, y, weights)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dataloaders.append(dataloader)

    return dataloaders

def compute_sample_weights(y, total_return):
    """
    Compute the sample weights for the given dataset. The sample weights are used to balance the dataset. The samples
    are weighted based on the number of samples in each class. The weights are computed as the inverse of the
    """
    # Compute log return from percentage change
    log_returns = torch.log(y + 1)

    # Convert to absolute values
    abs_return = torch.abs(log_returns)

    # Compute the sample weights
    sample_weights = abs_return / total_return

    return sample_weights

def compute_total_return(dataset):
    """
    Compute the total return for the given dataset. The total return is used to compute the sample weights. The total
    return is computed as the sum of the absolute values of the log returns.

    Parameters
    ----------
    dataset : The dataset to be used for computing the total return.

    Returns
    -------
    Total return for the given dataset.
    """
    return sum(y.abs().sum() for _, y in dataset)
import torch
from torch.nn import BCEWithLogitsLoss
from Models.BaseModel import BaseModel
from Modeling.DatasetManager import DatasetManager, DatasetTuple
from Metrics.ProfitCalculator import ProfitCalculator


class WalkForwardTester:
    """

    """
    def  __init__(self, model: BaseModel, dataset_manager: DatasetManager):
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

    def train(self, epochs, learning_rate, verbose=False) -> None:
        """
        Trains the model using walk forward testing. It trains the model on the training data and validates it on the
        validation data.

        Parameters
        ----------
        epochs : Number of epochs to train the model.
        learning_rate : Learning rate to be used for training.
        verbose : Whether to print training information.
        """
        # Get the training and validation data
        train_data = self.dataset_manager.get_training_data(binary_classification=True)

        # Train the model
        train(self.model, train_data, epochs, learning_rate, verbose)

    def evaluate_on_train(self) -> tuple[float, float, float]:
        """
        Evaluate the model on the training data. It uses the training data to evaluate the model. The model returns
        three values: the profit, the accuracy and the loss.

        Returns
        -------
        Profit, accuracy and loss of the model on the training data.
        """
        # Get the training and validation data
        train_data = self.dataset_manager.get_training_data(binary_classification=False)

        evaluation = evaluate(self.model, train_data)

        return evaluation

    def evaluate_on_test(self) -> tuple[float, float, float]:
        """
        Evaluate the model on the next given month. It uses the test data to evaluate the model.

        Returns
        -------
        Profit, accuracy and loss of the model on the test data.
        """
        # Get the training, validation and test data
        test_data = self.dataset_manager.get_test_data()

        evaluation = evaluate(self.model, test_data)

        return evaluation


def train(model: BaseModel, train_data: list[DatasetTuple], epochs: int, learning_rate: float,
          verbose=True) -> None:
    """
    Train the model using the given training data. It uses the BCEWithLogitsLoss function to compute the loss and
    Adam optimizer to optimize the model. The model is trained for the given number of epochs. This training scheme
    performs multiple epoch per batch instead of multiple batch per epoch. This is done to minimize data leakage
    in the train set, which the model may exploit to overfit the data.

    Parameters
    ----------
    model : The model to be trained.
    train_data : The training data to be used for training the model.
    epochs : Number of epochs to train the model.
    learning_rate : Learning rate to be used for training the model.
    verbose : Whether to print training information.
    """

    # Initialize the optimizer
    loss = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    average_loss = 0
    total_samples = 0
    for i, (x, y, z) in enumerate(train_data):
        x = x.to(model.device)
        y = y.to(model.device)
        z = z.to(model.device)
        total_loss = 0
        for epoch in range(epochs):
            # Forward pass
            y_pred = model(x, z)
            loss_value = loss(y_pred, y)
            total_loss = loss_value.item()

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        average_loss += total_loss * len(y)
        total_samples += len(y)

        if verbose:
            print(f"Batch {i+1}/{len(train_data)} has average Loss of : {average_loss/total_samples:.4f}")

def evaluate(model, dataset) -> tuple[float, float, float]:
    """
    Evaluate the model using the given dataset and metric. It also computes the accuracy of the model by comparing
    the sign of the predicted and actual values.

    Parameters
    ----------
    model : The model to be evaluated.
    dataset : The dataset to be evaluated.

    Returns
    -------
    The metric value and accuracy.
    """
    # Setup
    profit_calculator = ProfitCalculator()
    loss_fn = BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
        average_profit = 0
        average_accuracy = 0
        average_loss = 0
        total_samples = 0
        for X, y, z in dataset:
            X = X.to(model.device)
            y = y.to(model.device)
            z = z.to(model.device)
            y_pred = model(X, z)

            profit = profit_calculator(y_pred, y)

            # Accuracy checks if sign match
            signs = ((y_pred * y) > 0).float()
            accuracy = ((y_pred * y) > 0).float().mean()

            # Loss requires y to be binary
            y = torch.where(y > 0, 1, 0).float()
            loss = loss_fn(y_pred, y)

            average_profit += profit.item() * len(y)
            average_accuracy += accuracy.item() * len(y)
            average_loss += loss.item() * len(y)
            total_samples += len(y)

        average_profit /= total_samples
        average_accuracy /= total_samples
        average_loss /= total_samples

        return average_profit, average_accuracy, average_loss
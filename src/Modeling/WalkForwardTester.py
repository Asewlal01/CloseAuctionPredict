import torch
from torch.nn import BCEWithLogitsLoss
from Models.BaseModel import BaseModel
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager, convert_to_classification, DatasetTuple
from Modeling.DatasetManagers.ClosingDatasetManager import ExogenousDatasetTuple
import copy
import random


class WalkForwardTester:
    def  __init__(self, model: BaseModel, train_manager: IntradayDatasetManager, test_manager: IntradayDatasetManager,
                  sequence_size: int = 360, use_trading: bool = False, use_exogenous: bool = False) -> None:
        """
        Initializes the WalkForwardTester class.

        Parameters
        ----------
        model : Model to be tested
        train_manager : DatasetManager to be used for training
        test_manager : DatasetManager to be used for testing
        sequence_size : Size of the sequence to be used
        use_trading : Whether to use trading data
        use_exogenous : Whether to use exogenous data
        """
        self.model = model
        self.train_manager = train_manager
        self.test_manager = test_manager

        self.sequence_size = sequence_size
        self.use_trading = use_trading
        self.use_exogenous = use_exogenous

    def train(self, epochs, learning_rate, verbose=False, maximise_validation: bool=False,
              validation_size: float = 0.25) -> None:
        """
        Trains the model using walk forward testing. It trains the model on the training data and validates it on the
        validation data.

        Parameters
        ----------
        epochs : Number of epochs to train the model.
        learning_rate : Learning rate to be used for training.
        verbose : Whether to print training information.
        maximise_validation : Whether to use parameters where the validation metric is maximised.
        validation_size : Size of the validation set to be used for training.
        """
        # Get the training and validation data
        train_data = self.train_manager.get_dataset()

        # Train the model
        train(self.model, train_data, self.sequence_size, self.use_trading, self.use_exogenous,
              epochs, learning_rate, verbose,
              maximise_validation, validation_size)

    def evaluate_on_train(self) -> float:
        """
        Evaluate the model on the training data. It uses the training data to evaluate the model. The model returns
        three values: the profit, the accuracy and the loss.

        Returns
        -------
        Profit, accuracy and loss of the model on the training data.
        """
        # Get the training and validation data
        train_data = self.train_manager.get_dataset()

        evaluation = evaluate(self.model, train_data, self.sequence_size, self.use_trading, self.use_exogenous)

        return evaluation

    def evaluate_on_test(self) -> float:
        """
        Evaluate the model on the next given month. It uses the test data to evaluate the model.

        Returns
        -------
        Profit, accuracy and loss of the model on the test data.
        """
        # Get the training, validation and test data
        test_data = self.test_manager.get_dataset()

        evaluation = evaluate(self.model, test_data, self.sequence_size, self.use_trading, self.use_exogenous)

        return evaluation

    def test_predictions(self) -> list[torch.Tensor]:
        """
        Get all predictions on the test data.

        Returns
        -------
        List of predictions where each element represents the predictions for a given day and the true values
        """
        # Get the training, validation and test data
        test_data = self.test_manager.get_dataset()

        data = [
            process_sequence(*item, sequence_size=self.sequence_size, use_trading=self.use_trading,
                             use_exogenous=self.use_exogenous)
            for item in test_data
        ]

        # Get the predictions
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for item in data:
                if self.use_exogenous:
                    x, y, z = item
                    input_vars = [x, z]
                else:
                    x, y = item
                    input_vars = [x,]

                # Moving everything to the device
                input_vars = [var.to(self.model.device) for var in input_vars]

                # Forward pass
                y_pred = self.model(*input_vars)

                # Create 2D tensor with y_pred and y being the columns
                y_pred = y_pred.cpu()
                y_all = torch.column_stack([y_pred, y])
                predictions.append(y_all)

        return predictions

def train(model: BaseModel, train_data: list[DatasetTuple | ExogenousDatasetTuple],
          sequence_size: int, use_trading: bool, use_exogenous: bool,
          epochs: int, learning_rate: float, verbose: bool,
          maximise_validation: bool, validation_size: float) -> None:
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
    use_trading : Whether to use trading data.
    use_exogenous : Whether to use exogenous data.
    epochs : Number of epochs to train the model.
    learning_rate : Learning rate to be used for training the model.
    verbose : Whether to print training information.
    maximise_validation : Whether to use parameters where the validation metric is maximised.
    validation_size : Size of the validation set to be used for training.
    """

    # Initialize the loss function, optimizer and dataloaders
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = [
        process_sequence(*item, sequence_size=sequence_size, use_trading=use_trading, use_exogenous=use_exogenous)
        for item in train_data
    ]

    # Create validation set
    validation_data = None
    best_model = model.state_dict()
    best_score = float('inf')
    if maximise_validation:
        validation_index = int(validation_size * len(train_data))
        validation_data = train_data[-validation_index:]
        train_data = train_data[:-validation_index]

    model.train()
    for epoch in range(epochs):
        for item in train_data:
            # Unpacking the item
            if use_exogenous:
                x, y, z = item
                input_vars = [x, z]
            else:
                x, y = item
                input_vars = [x,]

            # Moving everything to the device
            input_vars = [var.to(model.device) for var in input_vars]
            y = y.to(model.device)

            # Training requires y to be binary
            y = convert_to_classification(y)

            # Forward pass
            y_pred = model(*input_vars)
            loss_per_sample = loss_fn(y_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss_per_sample.backward()
            optimizer.step()

        if verbose:
            train_loss = evaluate(model, train_data, sequence_size, True, use_exogenous)
            print(f"Epoch {epoch+1}/{epochs} has training Loss of : {train_loss:.4f}")

        if maximise_validation:
            # use_trading is set to true because we have already removed from the dataset at the beginning
            validation_loss = evaluate(model, validation_data, sequence_size, True, use_exogenous)
            if validation_loss < best_score:
                best_score = validation_loss
                best_model = copy.deepcopy(model.state_dict())

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} has validation Loss of : {validation_loss:.4f} \n")

        model.train()

    # Update with best model
    if maximise_validation:
        model.load_state_dict(best_model)

def process_sequence(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor=None, *,
                     sequence_size: int, use_trading: bool=False, use_exogenous: bool=False) -> DatasetTuple | ExogenousDatasetTuple:
    """
    Process the sequence of data. It removes the first sequence_size elements from the data. It also removes the
    Parameters
    ----------
    x : Input data
    y : Output data
    z : Exogenous data
    sequence_size : Size of the sequence to be used
    use_trading : Whether to use trading data
    use_exogenous : Whether to use exogenous data

    Returns
    -------
    Processed sequence of data.
    """
    x = x[:, -sequence_size:, :]
    if not use_trading:
        x = x[:, :, :22]

    return (x, y, z) if use_exogenous else (x, y)

def evaluate(model: BaseModel, data: list[DatasetTuple | ExogenousDatasetTuple], sequence_size: int,
             use_trading: bool, use_exogenous: bool) -> float:
    """
    Evaluate the model using the given dataset and metric. It also computes the accuracy of the model by comparing
    the sign of the predicted and actual values.

    Parameters
    ----------
    model : The model to be evaluated.
    data : The dataset to be used for evaluation.
    sequence_size : The size of the sequence to be used.
    use_trading : Whether to use trading data.
    use_exogenous : Whether to use exogenous data.

    Returns
    -------
    The metric value and accuracy.
    """
    # Setup
    loss_fn = BCEWithLogitsLoss()

    # Process the data
    data = [
        process_sequence(*item, sequence_size=sequence_size, use_trading=use_trading, use_exogenous=use_exogenous)
        for item in data
    ]

    with torch.no_grad():
        average_loss = 0
        model.eval()
        for item in data:
            # Unpacking the item
            if use_exogenous:
                x, y, z = item
                input_vars = [x, z]
            else:
                x, y = item
                input_vars = [x,]

            # Moving everything to the device
            input_vars = [var.to(model.device) for var in input_vars]
            y = y.to(model.device)

            # Loss requires y to be binary
            y = convert_to_classification(y)

            # Forward pass
            y_pred = model(*input_vars)
            loss = loss_fn(y_pred, y)
            average_loss += loss.item()

    return average_loss / len(data)
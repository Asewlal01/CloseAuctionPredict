import torch
from torch.nn import BCEWithLogitsLoss
from Models.BaseModel import BaseModel
from Modeling.DatasetManagers.IntradayDatasetManager import IntradayDatasetManager, convert_to_classification, DatasetTuple
from Modeling.DatasetManagers.ClosingDatasetManager import ExogenousDatasetTuple
from Metrics.ProfitCalculator import ProfitCalculator


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

        self.metric = ProfitCalculator()

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

    def evaluate_on_train(self) -> tuple[float, float, float]:
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

    def evaluate_on_test(self) -> tuple[float, float, float]:
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


def train(model: BaseModel, train_data: list[DatasetTuple | ExogenousDatasetTuple], sequence_size: int,
          use_trading: bool, use_exogenous: bool,
          epochs: int, learning_rate: float, verbose, maximise_validation: bool, validation_size: float) -> None:
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
    verbose : Whether to print training information.
    maximise_validation : Whether to use parameters where the validation metric is maximised.
    validation_size : Size of the validation set to be used for training.
    """

    # Initialize the loss function, optimizer and dataloaders
    loss = BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = [
        process_sequence(*item, sequence_size=sequence_size, use_trading=use_trading, use_exogenous=use_exogenous)
        for item in train_data
    ]

    # Create validation set
    if maximise_validation:
        validation_index = int(validation_size * len(train_data))
        validation_data = train_data[-validation_index:]
        train_data = train_data[:-validation_index]

        # Saving best score and model
        best_model = None
        best_score = float('inf')

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for item in train_data:
            # Unpacking the item
            if use_exogenous:
                x, y, z = item
                input_vars = [x, z]
            else:
                x, y = item
                input_vars = [x,]

            # Weights for each sample
            total_return = y.abs().sum()
            weights = compute_sample_weights(y, total_return)

            # Moving everything to the device
            input_vars = [var.to(model.device) for var in input_vars]
            y = y.to(model.device)
            weights = weights.to(model.device)

            # Training requires y to be binary
            y = convert_to_classification(y)

            # Forward pass
            y_pred = model(*input_vars)
            loss_per_sample = loss(y_pred, y)
            weighted_loss = weights * loss_per_sample
            loss_value = weighted_loss.sum()
            total_loss += loss_value.item()

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        if verbose:
            average_loss = total_loss / len(train_data)
            print(f"Epoch {epoch+1}/{epochs} has training Loss of : {average_loss:.4f}")

        if maximise_validation:
            # use_trading is set to true because we have already removed from the dataset at the beginning
            validation_loss = evaluate(model, validation_data, sequence_size, True, use_exogenous)
            if validation_loss < best_score:
                best_score = validation_loss
                best_model = model.state_dict()

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
        x = x[:, :, :20]

    return (x, y, z) if use_exogenous else (x, y)

def evaluate(model: BaseModel, data: list[DatasetTuple | ExogenousDatasetTuple], sequence_size: int,
             use_trading: bool, use_exogenous: bool) -> tuple[float, float, float]:
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
    loss_fn = BCEWithLogitsLoss(reduction='none')
    model.eval()

    # Process the data
    data = [
        process_sequence(*item, sequence_size=sequence_size, use_trading=use_trading, use_exogenous=use_exogenous)
        for item in data
    ]

    with torch.no_grad():
        average_loss = 0
        for item in data:
            # Unpacking the item
            if use_exogenous:
                x, y, z = item
                input_vars = [x, z]
            else:
                x, y = item
                input_vars = [x,]

            # Weights for each sample
            total_return = y.abs().sum()
            weights = compute_sample_weights(y, total_return)

            # Moving everything to the device
            input_vars = [var.to(model.device) for var in input_vars]
            y = y.to(model.device)
            weights = weights.to(model.device)

            # Forward pass
            y_pred = model(*input_vars)

            # Loss requires y to be binary
            y = convert_to_classification(y)
            loss = loss_fn(y_pred, y)
            loss = loss * weights
            loss = loss.sum()

            average_loss += loss.item()

        # Sample based
        average_loss /= len(data)

        return average_loss

def compute_sample_weights(y, total_return):
    """
    Compute the sample weights for the given dataset. The sample weights are used to balance the dataset. The samples
    are weighted based on the number of samples in each class. The weights are computed as the inverse of the
    """
    # Convert to absolute values
    abs_return = torch.abs(y)

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
    # Compute absolute returns
    abs_returns = [returns.abs() for _, returns in dataset]

    return sum(returns.sum() for returns in abs_returns)
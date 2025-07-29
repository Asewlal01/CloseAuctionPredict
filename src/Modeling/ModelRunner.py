import torch
from Models.BaseModel import BaseModel
from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager, BaseSampleTuple
from copy import deepcopy

DatasetTuple = tuple[list[BaseSampleTuple], list[BaseSampleTuple], list[BaseSampleTuple]]

# The indices related to the features in the dataset
LOB_FEATURE_INDICES = list(range(22))  # First 22 features are LOB features
TRADING_FEATURE_INDICES = list(range(22, 29))  # Next 7 features are trading features

class ModelRunner:
    """
    Class that manages the execution of a model on a dataset. Includes methods for training the model and evaluating its
    on the train, test and validation datasets.
    """
    def __init__(self, model: BaseModel, dataset_manager: BaseDatasetManager,
                 sequence_size: int=420, use_trading: bool=False, use_exogenous: bool=False,
                 horizon: int=0):
        """
        Initialize the ModelRunner.

        Parameters
        ----------
        model : The model to be trained and/or evaluated.
        dataset_manager : The dataset manager that provides the datasets.
        sequence_size : The size of the sequences to be used for training and evaluation. Default is 420.
        use_trading : Whether to use trading features in the model. Default is True.
        use_exogenous : Whether to use exogenous features in the model. Default is False.
        horizon : The horizon for the prediction. Default is 0, which means no horizon.
        """

        self.model = model
        self.dataset_manager = dataset_manager

        self.train_dataset, self.validation_dataset, self.test_dataset = self.get_datasets(sequence_size,
                                                                                           use_trading, use_exogenous,
                                                                                           horizon)

    def get_datasets(self, sequence_size: int, use_trading: bool, use_exogenous: bool, horizon: int) -> DatasetTuple:
        """
        Get the datasets for training, validation, and testing.

        Parameters
        ----------
        sequence_size : The size of the sequences to be used for training and evaluation.
        use_trading : Whether to use trading features in the model.
        use_exogenous : Whether to use exogenous features in the model.
        horizon : The horizon for the prediction. Default is 0, which means no horizon.

        Returns
        -------
        Dataset for each of the train, validation, and test sets.
        """
        # Getting the datasets
        train_dataset = self.dataset_manager.get_train_dataset()
        validation_dataset = self.dataset_manager.get_validation_dataset()
        test_dataset = self.dataset_manager.get_test_dataset()

        # Filtering each dataset based on the use of trading and exogenous features
        train_dataset = filter_features(train_dataset, sequence_size, use_trading, use_exogenous, horizon)
        validation_dataset = filter_features(validation_dataset, sequence_size, use_trading, use_exogenous, horizon)
        test_dataset = filter_features(test_dataset, sequence_size, use_trading, use_exogenous, horizon)

        return train_dataset, validation_dataset, test_dataset

    def train(self, epochs: int, learning_rate: float,
              stopping_epochs: int, verbose: bool=False) -> None:
        """
        Train the model on the training dataset.

        Parameters
        ----------
        epochs : Number of epochs to train the model for.
        learning_rate : Learning rate for the optimizer.
        stopping_epochs : Number of epochs to wait before stopping the training if no improvement is seen.
        verbose : Whether to print training progress. Default is False.
        """

        # Obtain the training dataset and validation dataset
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset

        # Train the model
        train_model(self.model, train_dataset, validation_dataset, epochs, learning_rate, stopping_epochs, verbose)

    def evaluate_on_train(self) -> float:
        """
        Evaluate the model on the training dataset.

        Returns
        -------
        Evaluation loss of the model on the training dataset.
        """
        return evaluate_model(self.model, self.train_dataset, torch.nn.BCEWithLogitsLoss())

    def evaluate_on_validation(self) -> float:
        """
        Evaluate the model on the validation dataset.

        Returns
        -------
        Evaluation loss of the model on the validation dataset.
        """
        return evaluate_model(self.model, self.validation_dataset, torch.nn.BCEWithLogitsLoss())

    def evaluate_on_test(self) -> float:
        """
        Evaluate the model on the test dataset.

        Returns
        -------
        Evaluation loss of the model on the test dataset.
        """
        return evaluate_model(self.model, self.test_dataset, torch.nn.BCEWithLogitsLoss())

    def predictions_on_train(self):
        """
        Get the predictions of the model on the training dataset.

        Returns
        -------
        List of predictions for each sample in the training dataset.
        """
        return predictions_model(self.model, self.train_dataset)

    def predictions_on_validation(self):
        """
        Get the predictions of the model on the validation dataset.

        Returns
        -------
        List of predictions for each sample in the validation dataset.
        """
        return predictions_model(self.model, self.validation_dataset)

    def predictions_on_test(self):
        """
        Get the predictions of the model on the test dataset.

        Returns
        -------
        List of predictions for each sample in the test dataset.
        """
        return predictions_model(self.model, self.test_dataset)


def filter_features(dataset: list[BaseSampleTuple],
                    sequence_size: int, use_trading: bool, use_exogenous: bool,
                    horizon: int) -> list[BaseSampleTuple]:
    """
    Filter the features in the dataset based on the use of trading and exogenous features.

    Parameters
    ----------
    dataset : The dataset to filter.
    sequence_size : The size of the sequences to be fed into the model.
    use_trading : Whether to use trading features.
    use_exogenous : Whether to use exogenous features.
    horizon : The horizon for the prediction. Default is 0, which means no horizon.

    Returns
    -------
    Filtered dataset.
    """
    filtered_dataset = []
    for sample in dataset:
        x, y, *z = sample

        # First 22 features are lob, followed by trading features
        if not use_trading:
            x = x[:, :, LOB_FEATURE_INDICES]

        if horizon > 0:
            x = x[:, :-horizon]  # Remove the last horizon features from the input

        # We only use the last sequence_size features for training
        x = x[:, -sequence_size:]

        # If we are not using exogenous features, we remove them
        if not use_exogenous:
            z = []

        # Append the filtered sample to the dataset
        filtered_dataset.append((x, y, *z))

    return filtered_dataset

def train_model(model: BaseModel, train_dataset: list[BaseSampleTuple], validation_dataset: list[BaseSampleTuple],
                epochs: int, learning_rate: float, stopping_epochs: int,
                verbose=False) -> None:
    """
    Train the model on the provided dataset.

    Parameters
    ----------
    model : The model to be trained.
    train_dataset : The dataset to train on.
    validation_dataset : The dataset to validate the model on.
    epochs : Number of epochs to train the model for.
    learning_rate : Learning rate for the optimizer.
    stopping_epochs : Number of epochs to wait before stopping the training if no improvement is seen.
    verbose : Whether to print training progress. Default is False.
    """
    # Define the optimizer and loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Best validation loss and state
    best_loss = float('inf')
    best_state = deepcopy(model.state_dict())
    counter = 0

    # Training loop
    for epoch in range(epochs):
        train_single_epoch(model, train_dataset, loss_fn, optimizer)

        # Get the loss on validation dataset
        validation_loss = evaluate_model(model, validation_dataset, loss_fn)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        # Print the results is verbose is True
        if verbose:
            train_evaluation = evaluate_model(model, train_dataset, loss_fn)
            print(f'Epoch {epoch + 1}/{epochs}: \n Train Loss: {train_evaluation:.4f}, Validation Loss: {validation_loss:.4f}')

        if counter >= stopping_epochs:
            if verbose:
                print(f'Stopping training after {stopping_epochs} epochs without improvement.')
            break

    # Load the best state
    model.load_state_dict(best_state)


def train_single_epoch(model: BaseModel, train_dataset: list[BaseSampleTuple],
                 loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """
    This function trains for one epoch on the provided dataset

    Parameters
    ----------
    model : The model to be trained.
    train_dataset : The dataset to train on.
    loss_fn : The loss function to use for training.
    optimizer : The optimizer to use for training.
    """
    model.train()
    for x, y, *z in train_dataset:
        # Backward pass and optimization
        optimizer.zero_grad()

        # Move to same device as model
        x, y, z = move_to_device(model, x, y, z)

        # Forward pass
        output = model(x, *z)

        # Y should be binary for classification tasks
        y = convert_to_classification(y)

        # Compute the loss
        loss =  loss_fn(output, y)

        # Backward pass
        loss.backward()
        optimizer.step()

def convert_to_classification(y) -> torch.Tensor:
    """
    Convert the output of the dataset to a binary classification problem. The output is 1 if the value is greater than 0
    and 0 otherwise. This is used for binary classification problems.
    Parameters
    ----------
    y : Output variable to convert. The output is assumed to be a torch tensor.

    Returns
    -------
    Binary classification output.
    """

    return torch.where(y > 0, 1, 0).float()

def evaluate_model(model: BaseModel, dataset: list[BaseSampleTuple], loss_fn: torch.nn.Module) -> float:
    """
    Evaluate the model on the provided dataset. This function should be implemented in subclasses to provide specific

    Parameters
    ----------
    model : The model to be evaluated.
    dataset : The dataset to evaluate the model on.
    loss_fn : The loss function to use for evaluation.

    Returns
    -------
    Evaluation loss of the model on the dataset
    """
    # Initialize the total loss and number of samples
    total_loss = 0
    samples = 0

    model.eval()
    with torch.no_grad():
        for x, y, *z in dataset:
            # Move to same device as model
            x, y, z = move_to_device(model, x, y, z)

            # Forward pass
            output = model(x, *z)

            # Y should be binary for classification tasks
            y = convert_to_classification(y)

            # Compute the loss
            loss = loss_fn(output, y)
            total_loss += loss.item() * len(y)
            samples += len(y)

    return total_loss / samples

def predictions_model(model: BaseModel, dataset: list[BaseSampleTuple]) -> list[torch.Tensor]:
    """
    Get the predictions of the model on the provided dataset.

    Parameters
    ----------
    model : The model to be evaluated.
    dataset : The dataset to evaluate the model on.

    Returns
    -------
    List of predictions for each sample in the dataset.
    """
    # List for saving the predictions
    predictions = []

    model.eval()
    with torch.no_grad():
        for x, y, *z in dataset:
            x = x.to(model.device)
            z = [feature.to(model.device) for feature in z]
            output = model(x, *z)

            # Bring output to cpu
            output = output.cpu()

            # Combine to a single tensor
            y_all = torch.column_stack([output, y])
            predictions.append(y_all)

    return predictions

def move_to_device(model: BaseModel, x: torch.Tensor, y: torch.Tensor, z: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Move the model to the specified device.

    Parameters
    ----------
    model : The model to be moved.
    x : Input tensor.
    y : Output tensor.
    z : Exogenous features tensor.
    """
    x = x.to(model.device)
    y = y.to(model.device)
    z = [feature.to(model.device) for feature in z]

    return x, y, z
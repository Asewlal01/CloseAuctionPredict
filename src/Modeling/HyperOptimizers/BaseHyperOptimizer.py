from Modeling.DatasetManagers.ClosingDatasetManager import ClosingDatasetManager
from Modeling.ModelRunner import ModelRunner
from Models.BaseModel import BaseModel
import optuna, os, torch, gc
import yaml

# Constants for sequence length
MIN_SEQUENCE_LENGTH = 30
MAX_SEQUENCE_LENGTH = 420
LENGTH_STEP_SIZE = 10

# Constants for dataset sizes
TRAIN_SIZE = 189
VALIDATION_SIZE = 20

# Training parameters
EPOCHS = 100
LEARNING_RATE = 1e-3

# Path to the config file
CONFIG_PATH = 'configs/tuning_config.yaml'


class BaseHyperOptimizer:
    """
    Class to optimize the hyperparameters of the model using Optuna.
    """

    def __init__(self, path_to_datasets: str, n_sets: int = 3, prefix: str = 'set_'):
        """
        Initialize the hyperparameter optimizer.

        Parameters
        ----------
        path_to_datasets : Path to the directory containing the datasets.
        n_sets : Number of sets to use for walk-forward validation. Default is 3.
        prefix : Prefix for the dataset directories. Default is 'set_'.
        """

        self.path_to_datasets = path_to_datasets
        self.n_sets = n_sets
        self.prefix = prefix

    def optimize(self, name: str, save_path: str, n_trials: int = 100):
        """
        Optimize the hyperparameters of the model.

        Parameters
        ----------
        name : The name of the study.
        save_path : The path to save the results.
        n_trials : The number of trials to run.  Default is 100.
        """
        # Make sure the save path exists
        os.makedirs(save_path, exist_ok=True)


        storage = f'sqlite:///{save_path}/results.db'
        study = optuna.create_study(direction='minimize',
                                    study_name=name,
                                    storage=storage,
                                    load_if_exists=True)

        # Optimize the objective function
        objective = self.generate_objective()
        study.optimize(objective, n_trials=n_trials)

        # Save the results
        df = study.trials_dataframe()
        save_results = os.path.join(save_path, f'{name}_results.csv')
        df.to_csv(save_results, index=False)

    def generate_objective(self) -> callable:
        """
        Generate the objective function for Optuna.

        Returns
        -------
        A function that takes a trial and returns the average loss.
        """

        def objective(trial):
            """
            Objective function for Optuna.

            Parameters
            ----------
            trial : The Optuna trial object.

            Returns
            -------
            Average loss over the walk-forward validation.
            """
            # Sequence size is needed to construct the model
            sequence_size = trial.suggest_int('sequence_size', MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH,
                                              step=LENGTH_STEP_SIZE)

            # Create the model using the model function
            model = self.generate_model(trial, sequence_size)

            # Initialize metric to optimize
            test_loss = 0

            for i in range(self.n_sets):
                path_to_set = os.path.join(self.path_to_datasets, f'{self.prefix}{i + 1}')

                # Initialize dataset managers
                dataset_manager = ClosingDatasetManager(path_to_set)
                dataset_manager.setup_dataset(TRAIN_SIZE, VALIDATION_SIZE)

                # Initialize model runner
                model_runner = ModelRunner(model, dataset_manager, sequence_size,
                                           use_trading=True, use_exogenous=True)

                # Training the model
                model_runner.train(EPOCHS, LEARNING_RATE)

                # Evaluate the model on the test set
                loss = model_runner.evaluate_on_test()
                test_loss += loss / self.n_sets

                # Reset the model parameters for the next iteration
                model.reset_parameters()

                # Clearing memory to avoid memory overflow
                clear_memory()

            return test_loss

        return objective

    def generate_model(self, trial: optuna.Trial, sequence_size: int) -> BaseModel:
        """
        Generate the model based on the trial and sequence size.

        Parameters
        ----------
        trial : The Optuna trial object.
        sequence_size : The size of the sequence.

        Returns
        -------
        The model instance created based on the trial parameters.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @classmethod
    def instantiate_from_config(cls):
        """
        Instantiate the hyperparameter optimizer from a configuration file.
        """

        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)['run']

        path_to_datasets = config['path_to_dataset']
        n_sets = config['n_sets']
        prefix = config['prefix']
        trials = config['trials']
        save_path = config['save_path']

        return cls(path_to_datasets=path_to_datasets, n_sets=n_sets, prefix=prefix), save_path, trials


def clear_memory():
    """
    Clear the memory by deleting all variables in the current scope.
    """
    gc.collect()
    torch.cuda.empty_cache()  # If using CUDA, clear the GPU memory as well
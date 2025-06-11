from Modeling.DatasetManagers.ClosingDatasetManager import ClosingDatasetManager
from Modeling.ModelRunner import ModelRunner
from Models.BaseModel import BaseModel
import optuna, os, torch, gc
import yaml

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
        self.config = None
        self.name = None

    def optimize(self, name: str=None, save_path: str=None, n_trials: int=None):
        """
        Optimize the hyperparameters of the model.

        Parameters
        ----------
        name : The name of the study.
        save_path : The path to save the results.
        n_trials : The number of trials to run.  Default is 100.
        """

        # Set default values for save_path, and n_trials if not provided
        save_path = save_path if save_path is not None else self.config['run']['save_path']
        n_trials = n_trials if n_trials is not None else self.config['run']['n_trials']

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
            min_sequence, max_sequence, step_sequence = self.config['common']['sequence_size']
            sequence_size = trial.suggest_int('sequence_size', min_sequence, max_sequence,
                                              step=step_sequence)

            # Create the model using the model function
            model = self.generate_model(trial, sequence_size)

            # Initialize metric to optimize
            test_loss = 0

            # Generate all dataset managers
            dataset_managers = []
            for i in range(self.n_sets):
                # Initialize dataset managers
                path_to_set = os.path.join(self.path_to_datasets, f'{self.prefix}{i + 1}')
                dataset_manager = ClosingDatasetManager(path_to_set)
                training_params = self.config['training_params']

                train_size = training_params['train_size']
                validation_size = training_params['validation_size']
                dataset_manager.setup_dataset(train_size, validation_size)
                dataset_managers.append(dataset_manager)

            for i in range(self.n_sets):
                # Initialize model runner
                dataset_manager = dataset_managers[i]
                model_runner = ModelRunner(model, dataset_manager, sequence_size,
                                           use_trading=True, use_exogenous=True)

                # Training the model
                training_params = self.config['training_params']
                epochs = training_params['epochs']
                learning_rate = training_params['learning_rate']
                model_runner.train(epochs, learning_rate)

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
            config = yaml.safe_load(f)

        # Get dataset configuration
        dataset_config = config['dataset']
        path_to_datasets = dataset_config['path_to_dataset']
        n_sets = dataset_config['n_sets']
        prefix = dataset_config['prefix']

        # Return object
        optimizer = cls(path_to_datasets, n_sets, prefix)
        optimizer.config = config

        return optimizer

    def generate_common_parameter(self, trial: optuna.Trial) -> tuple:
        """
        Generate parameters that are common to all models.

        Returns
        -------
        A tuple containing the number of neurons in each fully connected layer.
        """
        common_config = self.config['common']
        layers_min, layers_max, layers_step = common_config['fc_layers']
        neurons_min, neurons_max, neurons_step = common_config['fc_neurons']

        fc_neurons = []
        for layer in range(layers_min, layers_max + 1, layers_step):
            neurons = trial.suggest_int(f'fc_neurons_{layer}', neurons_min, neurons_max, step=neurons_step)
            fc_neurons.append(neurons)

        # Dropout
        dropout_min, dropout_max, dropout_step = common_config['dropout']
        dropout = trial.suggest_float('dropout', dropout_min, dropout_max, step=dropout_step)

        return fc_neurons, dropout


def clear_memory():
    """
    Clear the memory by deleting all variables in the current scope.
    """
    gc.collect()
    torch.cuda.empty_cache()  # If using CUDA, clear the GPU memory as well
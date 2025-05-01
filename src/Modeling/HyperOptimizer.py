import torch
from Modeling.DatasetManager import DatasetManager
from Modeling.WalkForwardTester import WalkForwardTester
import optuna


class HyperOptimizer:
    """
    Class to optimize the hyperparameters of the model using Optuna.
    """

    def __init__(self, dataset_manager: DatasetManager, start_month: str):
        """
        Initialize the hyperparameter optimizer.

        Parameters
        ----------
        dataset_manager : The dataset manager to use for loading the data.
        start_month : The starting month for the training data. Format: 'YYYY-MM'
        """
        self.dataset_manager = dataset_manager
        self.start_month = start_month

    def optimize(self, model_fn: callable, save_path: str, n_trials: int = 100):
        """
        Optimize the hyperparameters of the model.

        Parameters
        ----------
        model_fn : The function to create the model. Should take trail as an argument and return a model, number of
        epochs and learning rate.
        save_path : The path to save the results.
        n_trials : The number of trials to run.  Default is 100.
        """
        # Function to optimize
        def objective(trial):
            self.dataset_manager.setup_dataset(self.start_month)

            model, epochs, lr = model_fn(trial)
            model.to('cuda')

            # Setup tester
            tester = WalkForwardTester(model, self.dataset_manager)

            average_accuracy = 0
            for i in range(3):
                tester.train(epochs, lr, verbose=False)
                evaluation = tester.evaluate_on_test()

                # We only need the accuracy
                accuracy = evaluation[1]
                average_accuracy += accuracy * 1/3

                model.reset_parameters()
                self.dataset_manager.increment_dataset()

            return average_accuracy

        study = optuna.create_study(direction='maximize', study_name='hyperparameter_optimization')

        # Optimize the objective function
        study.optimize(objective, n_trials=n_trials)

        # Save the results
        df = study.trials_dataframe()
        df.to_csv(save_path, index=False)
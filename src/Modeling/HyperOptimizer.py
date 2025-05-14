from Modeling.DatasetManagers.BaseDatasetManager import BaseDatasetManager, split_month, increment_month
from Modeling.WalkForwardTester import WalkForwardTester
import optuna

class HyperOptimizer:
    """
    Class to optimize the hyperparameters of the model using Optuna.
    """

    def __init__(self, train_manager: BaseDatasetManager, test_manager: BaseDatasetManager, start_month: str,
                 months_to_evaluate: int = 1):
        """
        Initialize the hyperparameter optimizer.

        Parameters
        ----------
        train_manager : The dataset manager to be used for training.
        test_manager : The dataset manager to be used for testing.
        start_month : The starting month for the training data. Format: 'YYYY-MM'
        months_to_evaluate : The number of months to evaluate for the training data.
        """
        self.train_manager = train_manager
        self.test_manager = test_manager
        self.start_month = start_month
        self.months_to_evaluate = months_to_evaluate

    def optimize(self, model_fn: callable, name: str, save_path: str, n_trials: int = 100):
        """
        Optimize the hyperparameters of the model.

        Parameters
        ----------
        model_fn : The function to create the model. Should take trail as an argument and return a model, number of
        epochs, learning rate and batch size.
        name : The name of the study.
        save_path : The path to save the results.
        n_trials : The number of trials to run.  Default is 100.
        """
        # Starting month
        train_start_month = self.start_month

        # Testing is 1 month after training end
        train_length = self.train_manager.num_months
        year, month = split_month(train_start_month)
        test_year, test_month = increment_month(year, month, train_length)
        test_start_month = f'{test_year}-{test_month}'

        # Function to optimize
        def objective(trial):
            model, epochs, lr, batch_size, sequence_size = model_fn(trial)
            self.train_manager.setup_dataset(train_start_month)
            self.test_manager.setup_dataset(test_start_month)

            model.to('cuda')

            # Setup tester
            tester = WalkForwardTester(model, self.train_manager, self.test_manager, sequence_size)

            average_loss = 0
            i = 0
            while i < self.months_to_evaluate:
                tester.train(epochs, lr, batch_size, verbose=False)
                _, _, loss = tester.evaluate_on_test()
                average_loss += loss / self.months_to_evaluate

                model.reset_parameters()

                if i + 1 < self.months_to_evaluate:
                    # Increment the dataset
                    self.train_manager.increment_dataset()
                    self.test_manager.increment_dataset()

                i += 1

            # Clear the memory
            self.train_manager.clear_memory()
            self.test_manager.clear_memory()

            return average_loss


        storage = f'sqlite:///{save_path}/results.db'
        study = optuna.create_study(direction='minimize', study_name=name,
                                    storage=storage, load_if_exists=True)

        # Optimize the objective function
        study.optimize(objective, n_trials=n_trials)

        # Save the results
        df = study.trials_dataframe()
        df.to_csv(save_path, index=False)
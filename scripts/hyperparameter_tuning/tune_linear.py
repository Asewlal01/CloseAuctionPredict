from Modeling.HyperOptimizers.LinearHyperOptimizer import LinearHyperOptimizer

if __name__ == '__main__':
    optimizer, save_path, trials = LinearHyperOptimizer.instantiate_from_config()
    optimizer.optimize(name='linear_regression',
                       save_path=save_path,
                       n_trials=trials)
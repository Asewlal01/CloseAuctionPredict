from Modeling.HyperOptimizers.RecurrentHyperOptimizer import RecurrentHyperOptimizer

if __name__ == '__main__':
    optimizer = RecurrentHyperOptimizer.instantiate_from_config()
    optimizer.optimize(name='LSTM')
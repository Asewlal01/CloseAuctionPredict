from Modeling.HyperOptimizers.LinearHyperOptimizer import LinearHyperOptimizer

if __name__ == '__main__':
    optimizer = LinearHyperOptimizer.instantiate_from_config()
    optimizer.optimize(name='Linear')
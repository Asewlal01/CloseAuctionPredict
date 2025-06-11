from Modeling.HyperOptimizers.TransformerHyperOptimizer import TransformerHyperOptimizer

if __name__ == '__main__':
    optimizer = TransformerHyperOptimizer.instantiate_from_config()
    optimizer.optimize(name='Transformer')
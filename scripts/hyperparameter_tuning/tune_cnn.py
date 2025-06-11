from Modeling.HyperOptimizers.ConvolutionalHyperOptimizer import ConvolutionalHyperOptimizer

if __name__ == '__main__':
    optimizer = ConvolutionalHyperOptimizer.instantiate_from_config()
    optimizer.optimize(name='CNN')
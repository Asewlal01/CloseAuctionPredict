from cnn import run_training
from Models.LinearRegression.TradeLobLinear import TradeLobLinear

def get_model(sequence_size):
    model = TradeLobLinear(
        sequence_size
    )
    return model

if __name__ == '__main__':
    sequence_size = 120
    model = get_model(sequence_size)
    epochs = 100
    lr = 1e-3
    name = 'linear'
    run_training(model, epochs, lr, sequence_size, name)

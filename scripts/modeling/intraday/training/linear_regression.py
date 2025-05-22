from cnn import run_training
from Models.LinearRegression.LobLinear import LobLinear

def get_model(sequence_size):
    model = LobLinear(
        sequence_size
    )
    return model

if __name__ == '__main__':
    sequence_size = 120
    model = get_model(120)
    epochs = 100
    lr = 1e-4
    name = 'linear'
    run_training(model, epochs, lr, sequence_size, name)
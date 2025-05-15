from cnn import run_training
from Models.ReccurentModels.TradeLobLSTM import TradeLobLSTM

def get_model():
    model = TradeLobLSTM(
        hidden_size=32,
        lstm_size=3,
        fc_neurons=[128, 64],
        dropout=0.2,
    )
    model.to('cuda')
    return model

if __name__ == '__main__':
    model = get_model()
    sequence_size = 120
    epochs = 100
    lr = 1e-3
    name = 'lstm'
    run_training(model, epochs, lr, sequence_size, name)

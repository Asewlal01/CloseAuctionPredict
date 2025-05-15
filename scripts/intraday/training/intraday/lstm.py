from cnn import run_training
from Models.ReccurentModels.LobLSTM import LobLSTM

def get_model():
    model = LobLSTM(
        hidden_size=8*8,
        lstm_size=3,
        fc_neurons=[5*8, 2*8, 6*8],
        dropout=4*0.1,
    )
    model.to('cuda')
    return model

if __name__ == '__main__':
    model = get_model()
    sequence_size = 45*8
    epochs = 6 * 5
    lr = 1e-3
    name = 'lstm'
    run_training(model, epochs, lr, sequence_size, name)

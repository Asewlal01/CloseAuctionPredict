from cnn import run_training
from Models.ReccurentModels.ExogenousTradeLobLSTM import ExogenousTradeLobLSTM

def get_model():
    model = ExogenousTradeLobLSTM(
        hidden_size=32,
        lstm_size=2,
        fc_neurons=[128, 64],
        dropout=0.1,
    )
    model.to('cuda')
    return model

if __name__ == '__main__':
    model = get_model()
    sequence_size = 120
    epochs = 100
    lr = 1e-4
    name = 'lstm'
    run_training(model, epochs, lr, sequence_size, name)

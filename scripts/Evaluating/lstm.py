from Models.ReccurentModels.LimitOrderBookLSTM import LimitOrderBookLSTM
from scripts.Evaluating.cnn import main

if __name__ == '__main__':
    model = LimitOrderBookLSTM(
        hidden_size=64,
        lstm_size=2,
        fc_neurons=[32, 16],
        dropout=0.1
    )
    model.to('cuda')
    model_name = 'LSTM'
    main(model, model_name)
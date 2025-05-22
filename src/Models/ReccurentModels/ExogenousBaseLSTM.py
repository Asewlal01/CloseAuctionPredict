from Models.ExogenousBaseModel import ExogenousBaseModel
from Models.ReccurentModels.BaseLSTM import BaseLSTM


class ExogenousBaseLSTM(ExogenousBaseModel, BaseLSTM):
    def __init__(self, feature_size: int, hidden_size: int, lstm_size: int, fc_neurons: list[int],
                 dropout: float = 0.5):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.fc_neurons = fc_neurons
        self.dropout = dropout

        expected_dim = 3
        ExogenousBaseModel.__init__(self, expected_dim)

    def build_model(self):
        BaseLSTM.build_model(self)
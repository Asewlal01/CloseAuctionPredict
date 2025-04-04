from Models.BaseModel import BaseModel
from torch import nn
from Layers.LastOutputLSTM import LastOutputLSTM

class BaseLSTM(BaseModel):
    """
    Base class for all the convolutional Models. It inherits from the BaseModel class.
    """
    def __init__(self, feature_size: int, hidden_size: int, lstm_size: int, fc_neurons: list[int],
                 dropout: float = 0.5):
        """
        Initializes the LSTM Neural Network for predicting the Closing Price of a stock.

        Parameters
        ----------
        feature_size : Number of features in the input data
        hidden_size : Dimension of the hidden state of the LSTM
        lstm_size : Number of LSTM layers
        fc_neurons : Number of nodes in the fully connected layers. Can be a list of integers or a single integer.
        """

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.fc_neurons = fc_neurons
        self.dropout = dropout

        expected_dim = 3
        super(BaseLSTM, self).__init__(expected_dim)


    def build_model(self) -> None:
        """
        Build the model by adding all layers to self.layers.
        """
        self.fc_neurons = [self.fc_neurons] if type(self.fc_neurons) == int else self.fc_neurons

        # Add the LSTM layer
        self.layers.append(nn.LSTM(self.feature_size, self.hidden_size, self.lstm_size, batch_first=True))

        # Add the output state layer
        self.layers.append(LastOutputLSTM())

        # Dropout layer
        self.layers.append(nn.Dropout(self.dropout))

        self.output_dim = self.hidden_size


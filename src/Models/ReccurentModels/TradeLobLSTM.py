from Models.ReccurentModels.BaseLSTM import BaseLSTM

class TradeLobLSTM(BaseLSTM):
    """
    LSTM Neural Network for predicting the Closing Price of a stock using Limit Order Book (LOB) data.
    """

    def __init__(self,  hidden_size: int, lstm_size: int, fc_neurons: list[int], dropout: float = 0.5):
        """
        Initializes the LSTM Neural Network.

        Parameters
        ----------
        hidden_size : Dimension of the hidden state of the LSTM
        lstm_size : Number of LSTM layers
        fc_neurons : Number of nodes in the fully connected layers. Can be a list of integers or a single integer.
        """
        feature_size = 29
        super().__init__(feature_size,  hidden_size, lstm_size, fc_neurons, dropout)

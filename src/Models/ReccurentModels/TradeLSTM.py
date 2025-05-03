from Models.ReccurentModels.BaseLSTM import BaseLSTM

class TradeLSTM(BaseLSTM):
    """
    LSTM Neural Network for predicting the Closing Price of a stock using Trade data. Input data
    is assumed to be a 2D tensor with the columns representing the return and volume of the stock respectively. The
    rows are assumed to be the time steps. The output of the model is a 1D tensor representing the predicted closing
    price of the stock.
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
        feature_size = 6
        super(TradeLSTM, self).__init__(feature_size,  hidden_size, lstm_size, fc_neurons, dropout)

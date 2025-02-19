from models.BaseModel import BaseModel
from models.ConvolutionalModels.BaseConvolve import Unsqueeze
from torch import nn

class BaseLSTM(BaseModel):
    """
    Base class for all the convolutional models. It inherits from the BaseModel class.
    """
    def __init__(self, feature_size, hidden_size: int, lstm_size: int, fc_neurons: list[int]):
        """
        Initializes the LSTM Neural Network for predicting the Closing Price of a stock.

        Parameters
        ----------
        feature_size : Number of features in the input data
        hidden_size : Dimension of the hidden state of the LSTM
        lstm_size : Number of LSTM layers
        fc_neurons : Number of nodes in the fully connected layers. Can be a list of integers or a single integer.
        """
        super(BaseLSTM, self).__init__()

        # Saving the layers
        layers = []
        if type(fc_neurons) == int:
            fc_neurons = [fc_neurons]

        # Add the unsqueeze layer
        layers.append(Unsqueeze(False))

        # Add the LSTM layer
        layers.append(nn.LSTM(feature_size, hidden_size, lstm_size, batch_first=True))

        # Add the output state layer
        layers.append(OutputStateLSTM())

        # Add fully connected layers
        input_size = hidden_size
        for output_size in fc_neurons:
            # Add the layer
            layers.append(nn.Linear(input_size, output_size))
            # Add the activation function
            layers.append(nn.ReLU())
            # Update the input size
            input_size = output_size

        # Output layer
        layers.append(nn.Linear(input_size, 1))

        # Save the layers
        self.layers = nn.ModuleList(layers)

class OutputStateLSTM(nn.Module):
    """
    Layer designed to extract the output state from a LSTM layer. The LSTM layer returns (output, (h_n, c_n)) where
    only the output is needed for the task. More specifically, we only need the last output of the sequence.
    """
    def __init__(self, multiple_layers: int=1):
        super(OutputStateLSTM, self).__init__()
        self.multiple_layers = multiple_layers

    def forward(self, x):
        return x[0][:, -1]
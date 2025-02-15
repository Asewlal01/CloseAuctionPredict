import torch.nn as nn

class TradeSequential(nn.Module):
    """
    LSTM Neural Network for predicting the Closing Price of a stock using the Trade data. Input data
    is assumed to be a 2D tensor with the columns representing the return and volume of the stock respectively. The
    rows are assumed to be the time steps. The output of the model is a 1D tensor representing the predicted closing
    price of the stock.
    """

    def __init__(self, n_sequences: int, hidden_size: int, lstm_size: int, fc_size: list[int]):
        """
        Initializes the LSTM Neural Network.

        Parameters
        ----------
        n_sequences : Number of sequences in the input data
        hidden_size : Size of the hidden state of the LSTM
        lstm_size : Number of LSTM layers
        fc_size : Number of neurons in the fully connected layers. Can be a list of integers or a single integer.
        """
        super(TradeSequential, self).__init__()

        # Saving the layers
        layers = []
        if type(fc_size) == int:
            fc_size = [fc_size]

        # Add the LSTM layer
        layers.append(nn.LSTM(2, hidden_size, lstm_size, batch_first=True))

        # Layer to flatten the output
        layers.append(nn.Flatten())
        input_size = hidden_size * n_sequences

        # Add fully connected layers
        for output_size in fc_size:
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

    def forward(self, x):
        # LSTM layer
        x, _ = self.layers[0](x)

        for layer in self.layers[1:]:
            x = layer(x)
        return x

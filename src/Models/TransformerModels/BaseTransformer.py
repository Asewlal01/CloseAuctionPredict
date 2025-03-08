from Models.BaseModel import BaseModel
from torch import nn
import torch


class BaseTransformer(BaseModel):
    """
    Base class for all the transformer Models. It inherits from the BaseModel class.
    """
    def __init__(self, feature_size: int, sequence_size: int, embedding_size: int, num_heads: int, dropout: float,
                 dim_feedforward: int, num_layers: int, fc_neurons: list[int]):
        """
        Initializes the Transformer Network for predicting the Closing Price of a stock.

        Parameters
        ----------
        feature_size : Number of features in the input data
        sequence_size : Number of time steps in the input data
        embedding_size : Size of features after the embedding layer
        num_heads : Number of heads in the multi-head attention
        dropout : Dropout rate
        dim_feedforward : Dimensionality of the feedforward network model in the Encoder Layer
        num_layers : Number of sub-encoder-layers in the encoder
        fc_neurons : Neurons in each fully connected layer after the transformer
        """

        self.feature_size = feature_size
        self.sequence_size = sequence_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.fc_neurons = fc_neurons

        super(BaseTransformer, self).__init__(3)


    def build_model(self):
        # Embedding layer to higher dimension
        self.layers.append(nn.Linear(self.feature_size, self.embedding_size))

        # Positional encoding
        self.layers.append(PositionalEncoding(self.sequence_size, self.embedding_size))

        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(self.embedding_size, self.num_heads, self.dim_feedforward, self.dropout,
                                                   batch_first=True)
        # Encoder
        self.layers.append(nn.TransformerEncoder(encoder_layer, self.num_layers))

        # Get the last output
        self.layers.append(LastOutputTransformer())

        # Fully connected layers
        input_size = self.embedding_size
        for output_size in self.fc_neurons:
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.ReLU())
            input_size = output_size

        self.output_dim = input_size

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model. It adds positional information to the input data.
    """
    def __init__(self, sequence_size, embedding_size):
        """
        Initializes the positional encoding layer.

        Parameters
        ----------
        sequence_size : Sequence size of the input data
        embedding_size : Feature size of the input data after the embedding layer
        """
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding
        self.embedding_params = nn.Parameter(torch.randn(sequence_size, embedding_size))

    def forward(self, x):
        return x + self.embedding_params

class LastOutputTransformer(nn.Module):
    """
    Layer designed to extract the output state from a Transformer layer. The Transformer layer returns the output
    of all the time steps in the sequence, but we only need the output of the last time step.
    """
    def __init__(self):
        """
        Initializes the layer
        """
        super(LastOutputTransformer, self).__init__()

    def forward(self, x):
        return x[:, -1, :]
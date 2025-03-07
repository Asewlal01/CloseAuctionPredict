from models.BaseModel import BaseModel
from torch import nn
import torch


class BaseTransformer(BaseModel):
    """
    Base class for all the transformer models. It inherits from the BaseModel class.
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
        super(BaseTransformer, self).__init__(3)

        # Embedding layer to higher dimension
        self.layers.append(nn.Linear(feature_size, embedding_size))

        # Positional encoding
        self.layers.append(PositionalEncoding(sequence_size, embedding_size))

        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward, dropout, batch_first=True)

        # Encoder
        self.layers.append(nn.TransformerEncoder(encoder_layer, num_layers))

        # Get the last output
        self.layers.append(LastOutputTransformer())

        # Fully connected layers
        input_size = embedding_size
        for output_size in fc_neurons:
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.ReLU())
            input_size = output_size

        # Output layer
        self.layers.append(nn.Linear(input_size, 1))

        # Saving all the layers
        self.layers = nn.ModuleList(self.layers)

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
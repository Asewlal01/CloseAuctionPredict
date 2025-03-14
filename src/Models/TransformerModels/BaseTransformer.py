from Models.BaseModel import BaseModel
from torch import nn
from Layers.LastOutputTransformer import LastOutputTransformer
from Layers.PositionalEncoding import PositionalEncoding


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

        expected_dim = 3
        super(BaseTransformer, self).__init__(expected_dim)


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

        # Apply dropout
        self.layers.append(nn.Dropout(self.dropout))

        # Fully connected layers
        input_size = self.embedding_size
        for output_size in self.fc_neurons:
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.ReLU())
            input_size = output_size

        self.output_dim = input_size
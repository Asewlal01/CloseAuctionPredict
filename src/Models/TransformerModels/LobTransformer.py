from Models.TransformerModels.BaseTransformer import BaseTransformer

class LobTransformer(BaseTransformer):
    """
    Transformer Network for predicting Stock Prices using Limit Order Book (LOB) data.
    """
    def __init__(self, sequence_size: int, embedding_size: int, num_heads: int, dropout: float,
                 dim_feedforward: int, num_layers: int, fc_neurons: list[int]):
        """
        Initializes the Transformer Network for predicting the Closing Price of a stock using

        Parameters
        ----------
        sequence_size : Number of time steps in the input data
        embedding_size : Size of features after the embedding layer
        num_heads : Number of heads in the multi-head attention
        dropout : Dropout rate
        dim_feedforward : Dimensionality of the feedforward network model in the Encoder Layer
        num_layers : Number of sub-encoder-layers in the encoder
        fc_neurons : Neurons in each fully connected layer after the transformer
        """
        feature_size = 22
        super().__init__(feature_size, sequence_size, embedding_size, num_heads, dropout,
                                              dim_feedforward, num_layers, fc_neurons)
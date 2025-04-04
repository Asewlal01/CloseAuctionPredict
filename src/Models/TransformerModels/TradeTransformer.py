from Models.TransformerModels.BaseTransformer import BaseTransformer

class TradeTransformer(BaseTransformer):
    """
    Transformer Network for predicting the Closing Price of a stock using the Trade data.
    """
    def __init__(self, sequence_size: int, embedding_size: int, num_heads: int, dropout: float,
                 dim_feedforward: int, num_layers: int, fc_neurons: list[int], fc_dropout: float):
        """
        Initializes the Transformer Network for predicting the Closing Price of a stock using the Trade data.

        Parameters
        ----------
        sequence_size : Number of time steps in the input data
        embedding_size : Size of features after the embedding layer
        num_heads : Number of heads in the multi-head attention
        dropout : Dropout rate
        dim_feedforward : Dimensionality of the feedforward network model in the Encoder Layer
        num_layers : Number of sub-encoder-layers in the encoder
        fc_neurons : Neurons in each fully connected layer after the transformer
        fc_dropout : Dropout rate for the fully connected layers
        """

        # Trade data has 5 channels: Open, High, Low, Close, Volume
        feature_size = 2
        super().__init__(feature_size, sequence_size, embedding_size, num_heads, dropout,
                                              dim_feedforward, num_layers, fc_neurons)
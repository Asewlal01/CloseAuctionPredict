from Models.BaseModel import BaseModel
from Models.TransformerModels.BaseTransformer import BaseTransformer

class ExogenousBaseTransformer(BaseTransformer, BaseModel):
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
        BaseModel.__init__(self, expected_dim)

    def build_model(self):
        BaseTransformer.build_model(self)
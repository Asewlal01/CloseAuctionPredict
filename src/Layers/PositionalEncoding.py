from torch import nn
import torch

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding_params
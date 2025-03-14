import torch
from torch import nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]
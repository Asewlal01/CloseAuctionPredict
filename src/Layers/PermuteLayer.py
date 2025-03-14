from torch import nn
import torch

class PermuteLayer(nn.Module):
    """
    This layer permutes the dimensions of the input tensor.
    """
    def __init__(self, dims: tuple[int, ...]) -> None:
        """
        Initializes the PermuteLayer class.

        Parameters
        ----------
        dims : Tuple of integers representing the new order of the dimensions.
        """
        super(PermuteLayer, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)
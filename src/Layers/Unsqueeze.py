import torch

class Unsqueeze(torch.nn.Module):
    """
    Layer designed to unsqueeze a tensor. This is needed when the input is not given as a batch, which may cause
    errors in the forward pass. Dimensions go from (N,) to (1, N) for example.
    """

    def __init__(self, expected_dim: int):
        """
        Initializes the unsqueeze layer.

        Parameters
        ----------
        expected_dim : Expection dimension of the input tensor. This layer will add a dimension if the input tensor
        has a dimension of `expected_dim - 1`.

        """
        super(Unsqueeze, self).__init__()
        self.expected_dim = expected_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If only one sample is given, add a batch dimension
        if x.dim() == self.expected_dim - 1:
            return x.unsqueeze(0)
        return x


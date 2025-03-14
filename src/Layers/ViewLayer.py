from torch import nn
import torch


class ViewLayer(nn.Module):
    """
    Layer designed to reshape a tensor. This is needed because input is typically given as (batch, sequence, features).
    However, Linear layers require the input to be (batch, features).
    """
    def __init__(self):
        super(ViewLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, -1)